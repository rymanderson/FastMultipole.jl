# FGS R4 sequence replay benchmark (BRAINSTORM 021, gate 2).
#
# Replays the retained R4 lexicographic leaf sequence using the saved census
# (per-source tall-matrix shapes) and directed edge list (per-source target
# segments), with the production RHS semantics (`rhs += old` then `rhs -= new`,
# never a delta product). No FastMultipole solver code is touched: this prices
# candidate schedules before any production change.
#
# Modes
#   serial   – lexicographic loop, single-thread BLAS gemv per source leaf
#   rowpar   – persistent worker team; coordinator solves the diagonal block,
#              workers compute disjoint contiguous row tiles (column-major
#              streaming), accumulate in private scratch, scatter to owned rows
#   handoff  – same coordination machinery with zero-work payloads → µs/handoff
#   dag      – pull-DAG schedule analysis from the lower edges: unit and
#              byte-weighted critical paths, measured-cost recurrence C_i,
#              and a list-schedule simulation with backward-filler model
#   dagteam  – REAL split dual-layout executor (gate 2d): lower triangle
#              stored target-major, consumed as readiness-driven pulls over
#              the directed lower edges; upper triangle stored source-major,
#              backward products run as lower-priority filler tasks whose
#              segments are reduced into the next sweep's upper accumulator
#              at the sweep boundary. Run under numactl --interleave (task
#              ownership is dynamic, owner first-touch is undefined here).
#
# Precision: (default) F64 storage+state; --f32 = F32 storage, convert on
# load, F64 accumulate/state; --f32-full = F32 everywhere (storage, state,
# accumulate, LU) — no convert cost, but a different numerical experiment
# that must separately pass the independent accuracy evaluator.
#
# Usage:
#   julia -t4 benchmark/fgs_sequence_replay.jl --mode rowpar --sweeps 9 \
#         --census <gemv_census.csv> --edges <dependency_edges.csv> [--f32|--f32-full]
#
# House rule: local runs ≤ 4 threads. Full-scale numbers come from a
# single-socket HPC node with pinned threads (see the companion slurm script).

using LinearAlgebra, Random, Printf
using Base.Threads: Atomic, atomic_add!, nthreads, @spawn

# ---------------------------------------------------------------- CLI

function getarg(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i+1]
end
hasflag(flag) = flag in ARGS

const MODE    = getarg("--mode", "serial")
const NSWEEPS = parse(Int, getarg("--sweeps", "9"))
const CENSUS  = getarg("--census", "gemv_census.csv")
const EDGES   = getarg("--edges", "dependency_edges.csv")
const USE_F32 = hasflag("--f32")
const F32FULL = hasflag("--f32-full")
const SMALL_BYTES = parse(Int, getarg("--small-bytes", "262144"))  # ≤ this: coordinator-serial
const DAG_B = parse(Float64, getarg("--dag-bandwidth", "0.0"))     # GB/s for dag mode (0 = measure serial first)
const DAG_H = parse(Float64, getarg("--dag-handoff-us", "0.0"))    # µs per task handoff in dag mode
# first-touch policy for rowpar coefficient pages: "serial" (coordinator fills,
# reproduces the serial-fill/parallel-consume NUMA arm) or "owner" (each worker
# first-touches its own tile rows — consumer-aligned placement; sub-page tiles
# can still defeat ownership: verify with numastat on HPC, don't assume).
# For an interleave control, launch under `numactl --interleave=all`.
const FIRST_TOUCH = getarg("--first-touch", "serial")

# ---------------------------------------------------------------- load structure

struct LeafInfo
    n::Int                   # source strengths (diagonal block is n×n)
    m::Int                   # total target rows of the tall nonself matrix
    deps::Vector{Int}        # dependent (target) leaves, ascending
end

function load_structure(census_path, edges_path)
    clines = readlines(census_path)
    @assert startswith(clines[1], "leaf,")
    nleaf = length(clines) - 1
    n = zeros(Int, nleaf); m = zeros(Int, nleaf)
    for L in @view clines[2:end]
        f = split(L, ',')
        j = parse(Int, f[1]); n[j] = parse(Int, f[3]); m[j] = parse(Int, f[2])
    end
    deps = [Int[] for _ in 1:nleaf]
    elines = readlines(edges_path)
    @assert startswith(elines[1], "source_leaf,")
    for L in @view elines[2:end]
        f = split(L, ',')
        push!(deps[parse(Int, f[1])], parse(Int, f[2]))
    end
    foreach(sort!, deps)
    for j in 1:nleaf
        @assert sum(n[i] for i in deps[j]; init=0) == m[j] "row composition mismatch at leaf $j"
    end
    [LeafInfo(n[j], m[j], deps[j]) for j in 1:nleaf]
end

const LEAVES = load_structure(CENSUS, EDGES)
const NLEAF  = length(LEAVES)
const OFFSET = cumsum([0; [L.n for L in LEAVES]])          # global strength offsets
const NSTR   = OFFSET[end]
const BYTES_PER_SWEEP = sum(8 * L.m * L.n for L in LEAVES)

# per-source segment maps: local row ranges ↔ global rhs ranges
struct SegMap
    loc_start::Vector{Int}   # local start row of each segment (cumulative)
    glob_start::Vector{Int}  # global rhs start row of each segment
    len::Vector{Int}
end
function segmap(L::LeafInfo)
    ns = length(L.deps)
    loc = Vector{Int}(undef, ns); glob = Vector{Int}(undef, ns); len = Vector{Int}(undef, ns)
    off = 0
    for (k, i) in enumerate(L.deps)
        loc[k] = off + 1; glob[k] = OFFSET[i] + 1; len[k] = LEAVES[i].n
        off += LEAVES[i].n
    end
    SegMap(loc, glob, len)
end
const SEGS = [segmap(L) for L in LEAVES]

# ---------------------------------------------------------------- allocate replay state

const RNG = MersenneTwister(20260918)

precname(::Type{TM}, ::Type{TS}) where {TM,TS} =
    TM === Float64 ? "F64" : (TS === Float64 ? "F32conv" : "F32full")

@printf "R4 replay: %d leaves, %d strengths, %.3f GB coefficients/sweep, mode=%s f32=%s f32full=%s threads=%d\n" NLEAF NSTR BYTES_PER_SWEEP/1e9 MODE USE_F32 F32FULL nthreads()

# cheap deterministic fill (values are irrelevant to timing; avoids shared RNG
# state so owner-mode first touch can fill tiles in parallel)
@inline fillval(::Type{T}, i, jc, j) where T =
    T(((i * 2654435761 + jc * 40503 + j * 97) % 1024) / 1024 - 0.5)
function fillmat!(A::Matrix{T}, rows::UnitRange{Int}, j::Int) where T
    @inbounds for jc in axes(A, 2), i in rows
        A[i, jc] = fillval(T, i, jc, j)
    end
end

function alloc_matrices(::Type{T}) where T
    mats = [Matrix{T}(undef, L.m, L.n) for L in LEAVES]
    for (j, A) in enumerate(mats)
        fillmat!(A, axes(A, 1)[1]:size(A, 1), j)
    end
    mats
end

# diagonal blocks: diagonally dominant, cached LU (state precision TS)
function alloc_lus(::Type{TS}=Float64) where TS
    lus = Vector{LU{TS,Matrix{TS},Vector{Int}}}(undef, NLEAF)
    for (j, L) in enumerate(LEAVES)
        A = TS.(rand(RNG, L.n, L.n) .- 0.5)
        A += TS(L.n + 1.0) * I
        lus[j] = lu!(A)
    end
    lus
end

# ---------------------------------------------------------------- kernels

# same-precision tile: y[r] = A[r,:] * x  via BLAS on a strided view
@inline function tile_gemv!(y::AbstractVector{T}, A::Matrix{T}, r::UnitRange{Int}, x::AbstractVector{T}) where T
    mul!(view(y, 1:length(r)), view(A, r, :), x)
end

# Float32-storage tile: Float64 accumulate, convert on load
function tile_gemv!(y::AbstractVector{Float64}, A::Matrix{Float32}, r::UnitRange{Int}, x::AbstractVector{Float64})
    len = length(r); r1 = first(r)
    yv = view(y, 1:len)
    fill!(yv, 0.0)
    @inbounds for jc in axes(A, 2)
        s = x[jc]
        @simd for k in 1:len
            yv[k] = muladd(Float64(A[r1+k-1, jc]), s, yv[k])
        end
    end
    yv
end

# scatter `+= old` then `-= new` over the segments covered by local tile rows,
# then store new into old. r is the local row range owned by this worker.
function scatter_tile!(rhs::Vector{T}, old::Vector{T}, ynew::AbstractVector{T},
                       sm::SegMap, r::UnitRange{Int}) where T
    k = searchsortedlast(sm.loc_start, first(r))
    row = first(r)
    while row <= last(r)
        seg_end = sm.loc_start[k] + sm.len[k] - 1
        hi = min(seg_end, last(r))
        g = sm.glob_start[k] + (row - sm.loc_start[k])
        @inbounds @simd for t in 0:(hi-row)
            rhs[g+t] += old[row+t]
        end
        @inbounds @simd for t in 0:(hi-row)
            rhs[g+t] -= ynew[(row-first(r))+1+t]
        end
        @inbounds @simd for t in 0:(hi-row)
            old[row+t] = ynew[(row-first(r))+1+t]
        end
        row = hi + 1
        k += 1
    end
end

# ---------------------------------------------------------------- serial replay

function run_serial(::Type{TM}, ::Type{TS}) where {TM,TS}
    mats = alloc_matrices(TM)
    lus  = alloc_lus(TS)
    strengths = rand(RNG, TS, NSTR)
    rhs  = zeros(TS, NSTR)
    old  = [zeros(TS, L.m) for L in LEAVES]
    ybuf = zeros(TS, maximum(L.m for L in LEAVES))
    # prime old products (initialization pass, untimed)
    for j in 1:NLEAF
        x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
        tile_gemv!(old[j], mats[j], 1:LEAVES[j].m, x)
    end
    times = Float64[]
    for s in 1:NSWEEPS
        t0 = time_ns()
        for j in 1:NLEAF
            x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
            ldiv!(lus[j], x)                       # diagonal solve (in place)
            clamp!(x, TS(-1e3), TS(1e3))           # keep replay numerics bounded
            tile_gemv!(ybuf, mats[j], 1:LEAVES[j].m, x)
            scatter_tile!(rhs, old[j], ybuf, SEGS[j], 1:LEAVES[j].m)
        end
        push!(times, (time_ns() - t0) / 1e9)
    end
    report("serial($(precname(TM,TS)))", times)
end

# ---------------------------------------------------------------- persistent row-parallel replay

mutable struct TeamState
    epoch::Atomic{Int}        # bumped once per parallel leaf
    done::Atomic{Int}         # cumulative worker completions
    active::Base.RefValue{Int}
    stop::Base.RefValue{Bool}
end

function run_rowpar(::Type{TM}, ::Type{TS}; payload::Bool=true) where {TM,TS}
    nw = nthreads()                       # coordinator participates as worker 1
    lus  = payload ? alloc_lus(TS) : LU{TS,Matrix{TS},Vector{Int}}[]
    strengths = rand(RNG, TS, NSTR)
    rhs  = zeros(TS, NSTR)
    old  = [zeros(TS, L.m) for L in LEAVES]
    ybufs = [zeros(TS, maximum(L.m for L in LEAVES)) for _ in 1:nw]

    # precompute per-leaf worker row tiles (balanced contiguous ranges)
    tiles = Matrix{UnitRange{Int}}(undef, NLEAF, nw)
    parallel_leaf = falses(NLEAF)
    for j in 1:NLEAF
        m = LEAVES[j].m
        bytes = 8 * m * LEAVES[j].n
        parallel_leaf[j] = payload ? (bytes > SMALL_BYTES) : true
        for w in 1:nw
            lo = div((w-1)*m, nw) + 1; hi = div(w*m, nw)
            tiles[j, w] = lo:hi
        end
    end

    # coefficient allocation with the selected first-touch policy
    mats = Vector{Matrix{TM}}()
    if payload
        if FIRST_TOUCH == "owner"
            mats = [Matrix{TM}(undef, L.m, L.n) for L in LEAVES]
            Threads.@threads :static for w in 1:nw
                for j in 1:NLEAF
                    if parallel_leaf[j]
                        fillmat!(mats[j], tiles[j, w], j)      # worker-owned rows
                    elseif w == 1
                        fillmat!(mats[j], 1:LEAVES[j].m, j)    # coordinator-local
                    end
                end
            end
        else
            mats = alloc_matrices(TM)                           # serial first touch
        end
    end

    if payload   # prime old products
        for j in 1:NLEAF
            x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
            tile_gemv!(old[j], mats[j], 1:LEAVES[j].m, x)
        end
    end

    st = TeamState(Atomic{Int}(0), Atomic{Int}(0), Ref(0), Ref(false))

    worker = w -> begin
        my_epoch = 0
        while true
            while st.epoch[] == my_epoch
                st.stop[] && return
                GC.safepoint()
                ccall(:jl_cpu_pause, Cvoid, ())
            end
            my_epoch += 1
            j = st.active[]
            if payload
                r = tiles[j, w]
                if !isempty(r)
                    x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
                    y = tile_gemv!(ybufs[w], mats[j], r, x)
                    scatter_tile!(rhs, old[j], ybufs[w], SEGS[j], r)
                end
            end
            atomic_add!(st.done, 1)
        end
    end

    workers = [@spawn worker($w) for w in 2:nw]

    times = Float64[]
    nhandoff = 0
    for s in 1:NSWEEPS
        t0 = time_ns()
        for j in 1:NLEAF
            if payload
                x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
                ldiv!(lus[j], x)
                clamp!(x, TS(-1e3), TS(1e3))
            end
            if parallel_leaf[j]
                st.active[] = j
                target_done = st.done[] + (nw - 1)
                atomic_add!(st.epoch, 1)
                nhandoff += 1
                if payload   # coordinator takes tile 1
                    r = tiles[j, 1]
                    if !isempty(r)
                        x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
                        tile_gemv!(ybufs[1], mats[j], r, x)
                        scatter_tile!(rhs, old[j], ybufs[1], SEGS[j], r)
                    end
                end
                while st.done[] < target_done
                    GC.safepoint()
                    ccall(:jl_cpu_pause, Cvoid, ())
                end
            elseif payload   # tiny leaf: coordinator-serial
                x = view(strengths, OFFSET[j]+1:OFFSET[j+1])
                tile_gemv!(ybufs[1], mats[j], 1:LEAVES[j].m, x)
                scatter_tile!(rhs, old[j], ybufs[1], SEGS[j], 1:LEAVES[j].m)
            end
        end
        push!(times, (time_ns() - t0) / 1e9)
    end
    st.stop[] = true
    atomic_add!(st.epoch, 1)   # release spinners
    foreach(wait, workers)

    label = payload ? "rowpar($(precname(TM,TS)), team=$nw, touch=$FIRST_TOUCH)" : "handoff(team=$nw)"
    report(label, times)
    if !payload
        tot = sum(times)
        @printf "  handoffs/sweep = %d, avg handoff = %.3f µs  (budget < 5.8 µs)\n" NLEAF (tot / (NSWEEPS * NLEAF) * 1e6)
    else
        npar = count(parallel_leaf)
        @printf "  parallel leaves = %d/%d (small-bytes threshold %d)\n" npar NLEAF SMALL_BYTES
    end
end

# ---------------------------------------------------------------- reporting

function report(label, times)
    tmin, tmed = minimum(times), sort(times)[max(1, end ÷ 2)]
    @printf "%-28s sweeps=%d  min %.4f s  med %.4f s  → useful BW (min): %.2f GB/s\n" label length(times) tmin tmed (BYTES_PER_SWEEP / 1e9 / tmin)
    @printf "  projected 81-sweep coefficient-stream time: %.3f s (min) / %.3f s (med)\n" (81 * tmin) (81 * tmed)
end

# ---------------------------------------------------------------- pull-DAG analysis

function run_dag()
    # lower (j<i) predecessor lists and per-target pull bytes
    preds = [Int[] for _ in 1:NLEAF]
    pull_bytes = zeros(Int, NLEAF)
    for j in 1:NLEAF, i in LEAVES[j].deps
        if j < i
            push!(preds[i], j)
            pull_bytes[i] += 8 * LEAVES[j].n * LEAVES[i].n
        end
    end
    lower_total = sum(pull_bytes)
    upper_total = BYTES_PER_SWEEP - lower_total
    nedges = sum(length, preds)
    @printf "lower edges = %d, lower bytes = %d, upper bytes = %d\n" nedges lower_total upper_total

    # unit-weight and byte-weighted longest paths (leaves are topologically ordered by index)
    Cu = zeros(Int, NLEAF); Cb = zeros(Float64, NLEAF)
    for i in 1:NLEAF
        pu = 0; pb = 0.0
        for j in preds[i]
            pu = max(pu, Cu[j]); pb = max(pb, Cb[j])
        end
        Cu[i] = pu + 1
        Cb[i] = pb + pull_bytes[i]
    end
    @printf "unit longest path = %d of %d (expect 279)\n" maximum(Cu) NLEAF
    @printf "byte-weighted span = %d bytes; work/span = %.3f (expect 2.849)\n" maximum(Cb) (lower_total / maximum(Cb))

    # measured-cost recurrence: t_pull = pull_bytes/B + h, t_solve measured in-process
    B = DAG_B > 0 ? DAG_B * 1e9 : nothing
    if B === nothing
        println("(--dag-bandwidth not given: pass measured serial/rowpar GB/s to get timed spans)")
        return
    end
    h = DAG_H * 1e-6
    lus = alloc_lus()
    tsolve = zeros(NLEAF)
    x = rand(RNG, maximum(L.n for L in LEAVES))
    for j in 1:NLEAF          # measure each diagonal ldiv! (median of 5)
        xv = view(x, 1:LEAVES[j].n)
        ts = [(@elapsed ldiv!(lus[j], xv)) for _ in 1:5]
        tsolve[j] = sort(ts)[3]
    end
    C = zeros(NLEAF)
    for i in 1:NLEAF
        p = 0.0
        for j in preds[i]; p = max(p, C[j]); end
        C[i] = p + pull_bytes[i] / B + tsolve[i] + h
    end
    span = maximum(C)
    work = lower_total / B + sum(tsolve) + NLEAF * h
    @printf "measured-cost recurrence (B=%.1f GB/s, h=%.2f µs): span %.4f s/sweep, work %.4f s, work/span %.3f\n" DAG_B DAG_H span work (work/span)

    # list-schedule simulation with W workers, critical-path priority; backward as filler
    for W in (4, 8, 16, 32)
        makespan = simulate_schedule(preds, pull_bytes, tsolve, B, h, W)
        fwd_idle = W * makespan - work
        backward_t = upper_total / B                       # single-worker-equivalent backward work
        drain = max(0.0, backward_t - fwd_idle) / W        # what idle time can't absorb
        @printf "  W=%2d: forward makespan %.4f s/sweep, idle %.4f worker·s, backward drain +%.4f s → sweep %.4f s, 81-sweep %.2f s\n" W makespan fwd_idle drain (makespan + drain) (81 * (makespan + drain))
    end
end

function simulate_schedule(preds, pull_bytes, tsolve, B, h, W)
    # priority = downstream critical path
    nsucc = [Int[] for _ in 1:NLEAF]
    for i in 1:NLEAF, j in preds[i]; push!(nsucc[j], i); end
    prio = zeros(NLEAF)
    for i in NLEAF:-1:1
        p = 0.0
        for k in nsucc[i]; p = max(p, prio[k]); end
        prio[i] = p + pull_bytes[i] / B + tsolve[i] + h
    end
    indeg = [length(preds[i]) for i in 1:NLEAF]
    ready_time = zeros(NLEAF)                    # when all preds are done
    ready = [i for i in 1:NLEAF if indeg[i] == 0]
    wfree = zeros(W)
    finish = zeros(NLEAF)
    done = 0
    while done < NLEAF
        # pick highest-priority ready task; tie to earliest ready
        sort!(ready; by=i -> -prio[i])
        i = popfirst!(ready)
        w = argmin(wfree)
        t0 = max(wfree[w], ready_time[i])
        finish[i] = t0 + pull_bytes[i] / B + tsolve[i] + h
        wfree[w] = finish[i]
        done += 1
        for k in nsucc[i]
            indeg[k] -= 1
            ready_time[k] = max(ready_time[k], finish[i])
            indeg[k] == 0 && push!(ready, k)
        end
    end
    maximum(finish)
end

# ---------------------------------------------------------------- split dual-layout team executor (gate 2d)
#
# Executes the split recurrence for real: each leaf i becomes ready when all
# its lower predecessors have published (readiness counters over the directed
# lower edges), then one worker gathers predecessor strengths, streams the
# target-major lower matrix (one aggregated gemv), forms
# x_i = b_i - Lx - u_i and solves the cached diagonal LU. Backward tasks
# (source-major upper products q_j = U_j x_j^{s+1}) are queued after each
# solve and run as lower-priority filler; their source-private segments are
# reduced target-owned into the next sweep's upper accumulator at the sweep
# boundary (reduction time is included in the sweep time). The frozen u^s is
# never overwritten mid-sweep. Preserves the lexicographic iterate by
# induction on the lower edges; execution order differs, so results are
# mathematically equivalent, not bitwise.

function run_dagteam(::Type{TM}, ::Type{TS}) where {TM,TS}
    nw = nthreads()
    # lower predecessors / upper target lists from the directed edges
    preds = [Int[] for _ in 1:NLEAF]
    for j in 1:NLEAF, i in LEAVES[j].deps
        j < i && push!(preds[i], j)
    end
    uppers = [[i for i in LEAVES[j].deps if i < j] for j in 1:NLEAF]

    # target-major lower storage (n_i × Σ n_j), source-major upper (mup_j × n_j)
    ptot = [sum(LEAVES[j].n for j in preds[i]; init=0) for i in 1:NLEAF]
    mup  = [sum(LEAVES[i].n for i in uppers[j]; init=0) for j in 1:NLEAF]
    Lmat = [Matrix{TM}(undef, LEAVES[i].n, ptot[i]) for i in 1:NLEAF]
    Umat = [Matrix{TM}(undef, mup[j], LEAVES[j].n) for j in 1:NLEAF]
    for i in 1:NLEAF
        ptot[i] > 0 && fillmat!(Lmat[i], 1:LEAVES[i].n, i)
        mup[i]  > 0 && fillmat!(Umat[i], 1:mup[i], i + NLEAF)
    end
    @assert sum(sizeof(TM) * length(A) for A in Lmat) + sum(sizeof(TM) * length(A) for A in Umat) ==
            div(BYTES_PER_SWEEP * sizeof(TM), 8) "split storage does not tile the census bytes"

    # reduction map: target i receives (source j, offset into q_j)
    red = [Tuple{Int,Int}[] for _ in 1:NLEAF]
    for j in 1:NLEAF
        off = 0
        for i in uppers[j]
            push!(red[i], (j, off))
            off += LEAVES[i].n
        end
    end

    lus = alloc_lus(TS)
    strengths = rand(RNG, TS, NSTR)
    bconst = rand(RNG, TS, NSTR)
    u_acc  = zeros(TS, NSTR)
    q = [zeros(TS, mup[j]) for j in 1:NLEAF]
    xg = [zeros(TS, max(1, maximum(ptot))) for _ in 1:nw]
    yb = [zeros(TS, maximum(L.n for L in LEAVES)) for _ in 1:nw]

    # static priority = byte-weighted downstream critical path (lower graph)
    nsucc = [Int[] for _ in 1:NLEAF]
    for i in 1:NLEAF, j in preds[i]; push!(nsucc[j], i); end
    prio = zeros(NLEAF)
    for i in NLEAF:-1:1
        p = 0.0
        for k in nsucc[i]; p = max(p, prio[k]); end
        prio[i] = p + Float64(LEAVES[i].n) * ptot[i]
    end

    # scheduler state: queues + indeg mutated only under qlock; ready-set
    # scan-pop is fine (largest earliest cohort is single digits)
    indeg0 = [length(preds[i]) for i in 1:NLEAF]
    roots  = [i for i in 1:NLEAF if indeg0[i] == 0]
    indeg  = copy(indeg0)
    readyQ = Int[]; backQ = Int[]
    qlock  = Threads.SpinLock()
    ntasks = NLEAF + count(>(0), mup)
    ndone  = Atomic{Int}(0)
    epoch  = Atomic{Int}(0)
    stop   = Ref(false)

    pop_ready!(Q) = begin
        best = 1
        for k in 2:length(Q)
            prio[Q[k]] > prio[Q[best]] && (best = k)
        end
        t = Q[best]; Q[best] = Q[end]; pop!(Q); t
    end

    do_lower(w, i) = begin
        n = LEAVES[i].n
        off = 0
        xgv = xg[w]
        @inbounds for j in preds[i]
            copyto!(xgv, off + 1, strengths, OFFSET[j] + 1, LEAVES[j].n)
            off += LEAVES[j].n
        end
        yv = view(yb[w], 1:n)
        if ptot[i] > 0
            yv = tile_gemv!(yb[w], Lmat[i], 1:n, view(xgv, 1:ptot[i]))
        else
            fill!(yv, zero(TS))
        end
        xv = view(strengths, OFFSET[i]+1:OFFSET[i+1])
        @inbounds for k in 1:n
            xv[k] = bconst[OFFSET[i]+k] - yv[k] - u_acc[OFFSET[i]+k]
        end
        ldiv!(lus[i], xv)
        clamp!(xv, TS(-1e3), TS(1e3))
        lock(qlock)
        for k in nsucc[i]
            (indeg[k] -= 1) == 0 && push!(readyQ, k)
        end
        mup[i] > 0 && push!(backQ, i)
        unlock(qlock)
        atomic_add!(ndone, 1)
    end

    do_back(w, j) = begin
        xv = view(strengths, OFFSET[j]+1:OFFSET[j+1])
        tile_gemv!(q[j], Umat[j], 1:mup[j], xv)
        atomic_add!(ndone, 1)
    end

    drain(w) = begin
        while ndone[] < ntasks
            task = 0; isback = false
            lock(qlock)
            if !isempty(readyQ)
                task = pop_ready!(readyQ)
            elseif !isempty(backQ)
                task = pop!(backQ); isback = true
            end
            unlock(qlock)
            if task == 0
                GC.safepoint()
                ccall(:jl_cpu_pause, Cvoid, ())
            elseif isback
                do_back(w, task)
            else
                do_lower(w, task)
            end
        end
    end

    reduce_u!() = begin   # target-owned; runs while workers are parked
        fill!(u_acc, zero(TS))
        @inbounds for i in 1:NLEAF
            base = OFFSET[i]; n = LEAVES[i].n
            for (j, off) in red[i]
                qj = q[j]
                @simd for k in 1:n
                    u_acc[base+k] += qj[off+k]
                end
            end
        end
    end

    worker(w) = begin
        my_epoch = 0
        while true
            while epoch[] == my_epoch
                stop[] && return
                GC.safepoint()
                ccall(:jl_cpu_pause, Cvoid, ())
            end
            my_epoch += 1
            drain(w)
        end
    end
    workers = [@spawn worker($w) for w in 2:nw]

    # prime u^0 from the initial strengths (untimed)
    for j in 1:NLEAF
        mup[j] > 0 && do_back(1, j)
    end
    reduce_u!()

    times = Float64[]
    for s in 1:NSWEEPS
        # order matters: reset ndone first (parks any laggard on empty
        # queues), then rebuild queues, then wake the team
        t0 = time_ns()
        ndone[] = 0
        copyto!(indeg, indeg0)
        lock(qlock); empty!(readyQ); append!(readyQ, roots); empty!(backQ); unlock(qlock)
        atomic_add!(epoch, 1)
        drain(1)
        while ndone[] < ntasks   # laggard finishing its last task body
            GC.safepoint(); ccall(:jl_cpu_pause, Cvoid, ())
        end
        reduce_u!()              # boundary reduction, counted in sweep time
        push!(times, (time_ns() - t0) / 1e9)
    end
    stop[] = true
    atomic_add!(epoch, 1)
    foreach(wait, workers)

    @assert all(isfinite, strengths) "non-finite replay state"
    report("dagteam($(precname(TM,TS)), team=$nw)", times)
    @printf "  lower tasks = %d (roots %d, edges %d), backward tasks = %d, lower/upper bytes = %d / %d (F64-equiv)\n" NLEAF length(roots) sum(length, preds) count(>(0), mup) sum(8 * LEAVES[i].n * ptot[i] for i in 1:NLEAF) sum(8 * mup[j] * LEAVES[j].n for j in 1:NLEAF)
end

# ---------------------------------------------------------------- dispatch

BLAS.set_num_threads(1)
const TMAT   = (USE_F32 || F32FULL) ? Float32 : Float64
const TSTATE = F32FULL ? Float32 : Float64
if MODE == "serial"
    run_serial(TMAT, TSTATE)
elseif MODE == "rowpar"
    run_rowpar(TMAT, TSTATE)
elseif MODE == "handoff"
    run_rowpar(Float64, Float64; payload=false)
elseif MODE == "dag"
    run_dag()
elseif MODE == "dagteam"
    run_dagteam(TMAT, TSTATE)
else
    error("unknown --mode $MODE")
end
