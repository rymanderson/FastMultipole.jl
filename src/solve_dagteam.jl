#=##############################################################################
FastGaussSeidel sweep_order=:dagteam — split dual-layout pull-DAG executor.

Production implementation of the BRAINSTORM 021 gate-2d design
(fgs_acceleration_recommendation_20260918.md; measured in
benchmark/fgs_sequence_replay.jl --mode dagteam): the nonself operator is
split in lexicographic leaf order into a target-major LOWER triangle and a
source-major UPPER triangle. Each sweep:

 1. leaf i becomes ready when all lower predecessors have published their
    updated strengths (readiness counters over the directed lower edges);
    a worker then gathers predecessor strengths (ascending source order),
    streams the aggregated lower block in ONE single-threaded GEMV, forms
    x_i = b_i − (Lx)_i − u_i and solves the cached diagonal LU in place;
 2. after each leaf solve, its source-major backward product
    q_j = U_j x_j^{s+1} is queued and runs as lower-priority filler work;
 3. at the sweep boundary the q_j are reduced target-owned, in fixed
    ascending source order, into the next sweep's upper accumulator u^{s+1}.
    The frozen u^s is NEVER overwritten mid-sweep.

By induction on the lower edges each leaf reads exactly the lower new
strengths and frozen upper contribution the lexicographic recurrence
requires, so the iterate is mathematically equivalent to :lexicographic
(NOT bitwise: the pull aggregates all predecessors into one GEMV, and the
upper contribution is rebuilt from full products instead of incremental
old/new scatter). Results are deterministic at any thread count: fixed
per-row arithmetic order in every GEMV, fixed reduction order, and
execution order cannot affect values (each pull reads only finalized
predecessor strengths and the frozen u^s).

The production RHS invariant is preserved at outer-iteration boundaries:
after the inner sweeps, `right_hand_side = external + farfield − Lx − Ux`
at the new iterate (rebuilt from the last sweep's saved lower pulls and the
boundary-reduced upper accumulator), so the residual check, convergence
callback, delta bookkeeping, and relaxation path in solve! are unchanged.
u^0 is primed from the actual starting strengths (nonzero warm starts).

Worker teams are spawned per inner-sweep block and stopped before control
returns to the outer iteration — the workers spin (never yield) while
active, which would otherwise starve the threaded FMM farfield pass.
=###############################################################################

#------- precision-dispatched GEMV kernels -------#

# same-precision: BLAS
@inline dagteam_gemv!(y::AbstractVector{T}, A::Matrix{T}, x::AbstractVector{T}) where T = mul!(y, A, x)

# Float32 storage, Float64 state (:f32conv): convert on load, column-major
function dagteam_gemv!(y::AbstractVector{Float64}, A::Matrix{Float32}, x::AbstractVector{Float64})
    fill!(y, 0.0)
    @inbounds for jc in axes(A, 2)
        s = x[jc]
        @simd for k in eachindex(y)
            y[k] = muladd(Float64(A[k, jc]), s, y[k])
        end
    end
    return y
end

#------- construction -------#

# leaf LU cache re-factorized in the sweep state precision TS (used when
# TS != TF; pivoting may differ from the TF factorization — the dagteam
# iterate at reduced precision is certified by the independent evaluator,
# not by bit comparison)
function build_leaf_lu_cache_as(::Type{TS}, self_matrices::Matrices) where TS
    data = TS.(self_matrices.data)
    factors = map(eachindex(self_matrices.sizes)) do k
        m, n = self_matrices.sizes[k]
        matrix_range = get_matrix_range(self_matrices, k, m, n)
        lu!(reshape(view(data, matrix_range), m, n); check=true)
    end
    bytes = sizeof(data) + sum(sizeof(F.ipiv) for F in factors; init=0)
    return LeafLUCache{TS,eltype(factors)}(data, factors, 0.0, bytes)
end

"""
    build_dagteam_plan(precision, nonself_matrices, sorted_list, index_map,
                       source_tree, target_tree, strengths_by_leaf,
                       targets_by_branch, self_matrices, leaf_lu_cache;
                       nworkers=Threads.nthreads())

Build the `DagTeamPlan` for `sweep_order=:dagteam` by repacking the
source-major nonself matrices into split triangular storage (see
`DagTeamPlan`). `precision` selects coefficient/state types: `:f64`
(TM = TS = TF), `:f32conv` (TM = Float32, TS = TF, convert-on-load), or
`:f32full` (TM = TS = Float32, Float32 leaf LU solves; outer-iteration
bookkeeping and residual checks remain TF).

Requirements checked here (throws otherwise): strength and rhs leaf rows are
aligned and tile 1:n ascending; every direct target branch's row range tiles
whole solve leaves; no direct target branch contains its own source leaf; no
duplicate (source, target-leaf) blocks. The source-major `nonself_matrices`
are retained unchanged (other sweep orders and warm-start replay still use
them) — :dagteam therefore holds a second, split copy of the coefficients.

`nworkers` caps the sweep team size (coordinator + spawned workers ≤
`nworkers`); capped-out workers are never spawned, so they cannot poll or
join active-team barriers. Clamped to `1:Threads.nthreads()`. Completion is
an atomic task counter over a shared queue (not a fixed-arrival barrier), so
any team size drains the same queue without deadlock.

`idle_policy` selects what a worker does when both queues are empty:
`:spin` (default, production) busy-polls the queue lock; `:backoff` pauses
in a bounded exponential loop gated on the `qhint` atomic queue-length hint,
never touching the lock while the hint is zero (BRAINSTORM 021 Stage 2
waiting-policy probe). Both policies preserve dependency visibility (`qhint`
is updated under the same lock that publishes tasks), progress (the pause
bound is finite and the drain loop re-checks the completion counter), and
arithmetic ordering (scheduling never affects values — see above).
"""
function build_dagteam_plan(precision::Symbol, nonself_matrices::Matrices{TF},
        sorted_list::Vector{SVector{2,Int32}}, index_map::Vector{UnitRange{Int}},
        source_tree::Tree, target_tree::Tree,
        strengths_by_leaf::Vector{UnitRange{Int}},
        targets_by_branch::Vector{UnitRange{Int}},
        self_matrices::Matrices{TF}, leaf_lu_cache;
        nworkers::Integer=Threads.nthreads(),
        idle_policy::Symbol=:spin) where TF

    precision in (:f64, :f32conv, :f32full) || throw(ArgumentError(
        "dagteam_precision must be :f64, :f32conv, or :f32full (got $(repr(precision)))"))
    idle_policy in (:spin, :backoff) || throw(ArgumentError(
        "dagteam idle_policy must be :spin or :backoff (got $(repr(idle_policy)))"))
    TM = precision === :f64 ? TF : Float32
    TS = precision === :f32full ? Float32 : TF

    n_leaves = length(source_tree.leaf_index)

    #--- leaf geometry; strength rows must alias rhs rows ---#

    offset = Vector{Int}(undef, n_leaves + 1)
    offset[1] = 0
    for i in 1:n_leaves
        r = strengths_by_leaf[i]
        r == targets_by_branch[source_tree.leaf_index[i]] || throw(ArgumentError(
            "dagteam requires each leaf's strength rows to equal its rhs rows " *
            "(leaf $i: strengths $(r) vs rhs $(targets_by_branch[source_tree.leaf_index[i]]))"))
        first(r) == offset[i] + 1 || throw(ArgumentError(
            "dagteam requires leaf strength rows to tile 1:n ascending (leaf $i starts at $(first(r)), expected $(offset[i] + 1))"))
        offset[i + 1] = last(r)
    end
    nstr = offset[end]
    nof(i) = offset[i + 1] - offset[i]
    leaf_starts = [first(targets_by_branch[i_branch]) for i_branch in source_tree.leaf_index]

    #--- directed edges from the actual source/target ranges ---#

    preds = [Int[] for _ in 1:n_leaves]    # lower: j < i, block feeds target i
    uppers = [Int[] for _ in 1:n_leaves]   # upper: i < j, block of source j
    for j in 1:n_leaves
        for index in index_map[j]
            i_target, _ = sorted_list[index]
            rows = targets_by_branch[i_target]
            isempty(rows) && continue
            lo = max(searchsortedlast(leaf_starts, first(rows)), 1)
            hi = max(searchsortedlast(leaf_starts, last(rows)), 1)
            (first(rows) == offset[lo] + 1 && last(rows) == offset[hi + 1]) || throw(ArgumentError(
                "dagteam: direct target branch $i_target rows $rows do not tile whole solve leaves"))
            for i in lo:hi
                i == j && throw(ArgumentError(
                    "dagteam: direct target branch $i_target of source leaf $j contains the source leaf itself"))
                j < i ? push!(preds[i], j) : push!(uppers[j], i)
            end
        end
    end
    for i in 1:n_leaves
        sort!(preds[i]); sort!(uppers[i])
        allunique(preds[i]) && allunique(uppers[i]) || throw(ArgumentError(
            "dagteam: duplicate direct block for leaf $i — overlapping target branches in the direct list"))
    end

    #--- split storage + reduction map ---#

    ptot = [sum(nof(j) for j in preds[i]; init=0) for i in 1:n_leaves]
    mup = [sum(nof(i) for i in uppers[j]; init=0) for j in 1:n_leaves]
    Lmat = [Matrix{TM}(undef, nof(i), ptot[i]) for i in 1:n_leaves]
    Umat = [Matrix{TM}(undef, mup[j], nof(j)) for j in 1:n_leaves]

    # per-target column offsets (ascending preds), per-source row offsets
    # (ascending uppers) — the latter double as the reduction offsets
    colofs = Vector{Vector{Int}}(undef, n_leaves)
    upofs = Vector{Vector{Int}}(undef, n_leaves)
    for i in 1:n_leaves
        c = Vector{Int}(undef, length(preds[i]))
        acc = 0
        for (pos, j) in enumerate(preds[i])
            c[pos] = acc
            acc += nof(j)
        end
        colofs[i] = c
        u = Vector{Int}(undef, length(uppers[i]))
        acc = 0
        for (pos, k) in enumerate(uppers[i])
            u[pos] = acc
            acc += nof(k)
        end
        upofs[i] = u
    end
    red = [Tuple{Int,Int}[] for _ in 1:n_leaves]
    for j in 1:n_leaves   # ascending j → fixed reduction order per target
        for (pos, i) in enumerate(uppers[j])
            push!(red[i], (j, upofs[j][pos]))
        end
    end

    # repack: tall-matrix rows follow index_map segment order; within a
    # segment, rows follow the branch's ascending global rows, i.e. whole
    # leaves lo:hi in ascending order (asserted above)
    for j in 1:n_leaves
        mat, _ = get_matrix_vector(nonself_matrices, j)
        nj = nof(j)
        size(mat, 1) == 0 || size(mat, 2) == nj || throw(ArgumentError(
            "dagteam: nonself matrix of source leaf $j has $(size(mat, 2)) columns, expected $nj"))
        r0 = 0
        for index in index_map[j]
            i_target, _ = sorted_list[index]
            rows = targets_by_branch[i_target]
            isempty(rows) && continue
            lo = max(searchsortedlast(leaf_starts, first(rows)), 1)
            hi = max(searchsortedlast(leaf_starts, last(rows)), 1)
            for i in lo:hi
                ni = nof(i)
                if j < i
                    c0 = colofs[i][searchsortedfirst(preds[i], j)]
                    A = Lmat[i]
                    @inbounds for c in 1:nj, r in 1:ni
                        A[r, c0 + c] = TM(mat[r0 + r, c])
                    end
                else
                    rof = upofs[j][searchsortedfirst(uppers[j], i)]
                    A = Umat[j]
                    @inbounds for c in 1:nj, r in 1:ni
                        A[rof + r, c] = TM(mat[r0 + r, c])
                    end
                end
                r0 += ni
            end
        end
        r0 == size(mat, 1) || throw(ArgumentError(
            "dagteam repack covered $r0 of $(size(mat, 1)) rows for source leaf $j"))
    end

    #--- graph auxiliaries ---#

    nsucc = [Int[] for _ in 1:n_leaves]
    for i in 1:n_leaves, j in preds[i]
        push!(nsucc[j], i)
    end
    prio = zeros(n_leaves)                 # byte-weighted downstream critical path
    for i in n_leaves:-1:1                 # successors always have larger index
        p = 0.0
        for k in nsucc[i]
            p = max(p, prio[k])
        end
        prio[i] = p + Float64(nof(i)) * ptot[i]
    end
    indeg0 = [length(preds[i]) for i in 1:n_leaves]
    roots = [i for i in 1:n_leaves if indeg0[i] == 0]

    #--- state and scratch ---#

    x = zeros(TS, nstr)
    b = zeros(TS, nstr)
    u = zeros(TS, nstr)
    lsum = zeros(TS, nstr)
    q = [zeros(TS, mup[j]) for j in 1:n_leaves]
    external_rhs = zeros(TF, nstr)
    # team size = length(xg): dagteam_start_team! spawns length(xg)-1 workers
    # and dagteam_initialize! partitions over 1:length(xg)
    nw = clamp(Int(nworkers), 1, Threads.nthreads())
    max_ptot = max(1, maximum(ptot; init=1))
    max_n = max(1, maximum(nof(i) for i in 1:n_leaves; init=1))
    xg = [zeros(TS, max_ptot) for _ in 1:nw]
    yb = [zeros(TS, max_n) for _ in 1:nw]

    if TS === TF
        leaf_lu_cache isa LeafLUCache || throw(ArgumentError(
            "sweep_order=:dagteam requires cache_leaf_lu=true"))
        lus = leaf_lu_cache
    else
        lus = build_leaf_lu_cache_as(TS, self_matrices)
    end

    indeg = copy(indeg0)
    readyQ = Int[]; sizehint!(readyQ, n_leaves)
    backQ = Int[]; sizehint!(backQ, n_leaves)
    ntasks = n_leaves + count(>(0), mup)

    return DagTeamPlan{TM,TS,TF,typeof(lus)}(preds, nsucc, uppers, prio, indeg0,
        roots, ptot, mup, Lmat, Umat, red, offset, x, b, u, lsum, q,
        external_rhs, lus, xg, yb, indeg, readyQ, backQ, Threads.SpinLock(),
        Threads.Atomic{Int}(0), ntasks, idle_policy, Threads.Atomic{Int}(0),
        [DagWorkerStats() for _ in 1:nw], Ref(false))
end

#------- sweep executor -------#

# gather ascending predecessor strengths and stream the aggregated lower
# block; returns the pull view (length n_i) in worker w's scratch
@inline function dagteam_pull!(plan::DagTeamPlan{TM,TS}, w::Int, i::Int) where {TM,TS}
    n = plan.offset[i + 1] - plan.offset[i]
    xg = plan.xg[w]
    off = 0
    @inbounds for j in plan.preds[i]
        nj = plan.offset[j + 1] - plan.offset[j]
        copyto!(xg, off + 1, plan.x, plan.offset[j] + 1, nj)
        off += nj
    end
    yv = view(plan.yb[w], 1:n)
    if plan.ptot[i] > 0
        dagteam_gemv!(yv, plan.Lmat[i], view(xg, 1:plan.ptot[i]))
    else
        fill!(yv, zero(TS))
    end
    return yv
end

function dagteam_do_lower!(plan::DagTeamPlan{TM,TS}, w::Int, i::Int) where {TM,TS}
    base = plan.offset[i]
    n = plan.offset[i + 1] - base
    yv = dagteam_pull!(plan, w, i)
    xv = view(plan.x, base + 1:base + n)
    @inbounds for k in 1:n
        plan.lsum[base + k] = yv[k]                       # saved: Lx at this sweep
        xv[k] = plan.b[base + k] - yv[k] - plan.u[base + k]
    end
    ldiv!(plan.lus.factorizations[i], xv)
    # publish under the queue lock (its release fence publishes xv to the
    # worker that pops a newly ready successor)
    lock(plan.qlock)
    @inbounds for k in plan.nsucc[i]
        (plan.indeg[k] -= 1) == 0 && push!(plan.readyQ, k)
    end
    plan.mup[i] > 0 && push!(plan.backQ, i)
    plan.qhint[] = length(plan.readyQ) + length(plan.backQ)
    unlock(plan.qlock)
    Threads.atomic_add!(plan.ndone, 1)
    return nothing
end

# backward product q_j = U_j x_j at the source's current (post-solve) strengths
@inline function dagteam_do_back_product!(plan::DagTeamPlan, j::Int)
    xv = view(plan.x, plan.offset[j] + 1:plan.offset[j + 1])
    dagteam_gemv!(plan.q[j], plan.Umat[j], xv)
    return nothing
end

# target-owned boundary reduction, serial in fixed ascending source order
function dagteam_reduce_u!(plan::DagTeamPlan{TM,TS}) where {TM,TS}
    fill!(plan.u, zero(TS))
    @inbounds for i in eachindex(plan.red)
        base = plan.offset[i]
        n = plan.offset[i + 1] - base
        for (j, off) in plan.red[i]
            qj = plan.q[j]
            @simd for k in 1:n
                plan.u[base + k] += qj[off + k]
            end
        end
    end
    return nothing
end

# scan-pop the highest-priority ready pull (largest earliest cohorts are
# single digits at R4; O(|Q|) scan beats heap overhead)
@inline function dagteam_pop_ready!(Q::Vector{Int}, prio::Vector{Float64})
    best = 1
    @inbounds for k in 2:length(Q)
        prio[Q[k]] > prio[Q[best]] && (best = k)
    end
    t = Q[best]
    Q[best] = Q[end]
    pop!(Q)
    return t
end

# Aggregates (only while plan.collect_stats[]; per-worker, padded — no shared
# counters, and idle-streak lock churn is deliberately NOT timed per spin):
#   lockmgmt_ns — busy-path lock+pop time (acquisitions on the critical path);
#   idle_ns     — empty-pop streaks, timed at the streak boundaries only;
#   busy_*_ns   — task bodies (do_lower! includes its publish lock).
function dagteam_drain!(plan::DagTeamPlan, w::Int)
    backoff = plan.idle_policy === :backoff
    cs = plan.collect_stats[]
    st = plan.stats[w]
    idle_t0 = UInt64(0)   # nonzero while inside an empty-pop streak
    while plan.ndone[] < plan.ntasks
        task = 0
        isback = false
        t0 = (cs && idle_t0 == 0) ? time_ns() : UInt64(0)
        lock(plan.qlock)
        if !isempty(plan.readyQ)
            task = dagteam_pop_ready!(plan.readyQ, plan.prio)
        elseif !isempty(plan.backQ)
            task = pop!(plan.backQ)
            isback = true
        end
        task == 0 || (plan.qhint[] = length(plan.readyQ) + length(plan.backQ))
        unlock(plan.qlock)
        t0 == 0 || (st.lockmgmt_ns += time_ns() - t0)
        if task == 0
            if cs
                idle_t0 == 0 && (idle_t0 = time_ns())
                st.empty_pops += 1
            end
            if backoff
                # bounded exponential pause gated on the queue-length hint —
                # no lock traffic while empty; progress is guaranteed because
                # qhint is set under the same lock that publishes tasks and
                # the outer loop re-checks the completion counter
                delay = 32
                while plan.qhint[] == 0 && plan.ndone[] < plan.ntasks
                    for _ in 1:delay
                        ccall(:jl_cpu_pause, Cvoid, ())
                    end
                    GC.safepoint()
                    delay = min(delay << 1, 4096)
                end
            else
                GC.safepoint()
                ccall(:jl_cpu_pause, Cvoid, ())
            end
        else
            if cs && idle_t0 != 0
                st.idle_ns += time_ns() - idle_t0
                idle_t0 = UInt64(0)
            end
            t1 = cs ? time_ns() : UInt64(0)
            if isback
                dagteam_do_back_product!(plan, task)
                Threads.atomic_add!(plan.ndone, 1)
                cs && (st.busy_back_ns += time_ns() - t1; st.n_back += 1)
            else
                dagteam_do_lower!(plan, w, task)
                cs && (st.busy_lower_ns += time_ns() - t1; st.n_lower += 1)
            end
        end
    end
    cs && idle_t0 != 0 && (st.idle_ns += time_ns() - idle_t0)
    return nothing
end

#------- worker team lifecycle -------#

mutable struct DagTeamRuntime
    epoch::Threads.Atomic{Int}
    stop::Base.RefValue{Bool}
    tasks::Vector{Task}
end

function dagteam_worker!(plan::DagTeamPlan, rt::DagTeamRuntime, w::Int)
    my_epoch = 0
    while true
        while rt.epoch[] == my_epoch
            rt.stop[] && return nothing
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
        my_epoch += 1
        dagteam_drain!(plan, w)
    end
end

function dagteam_start_team!(plan::DagTeamPlan)
    rt = DagTeamRuntime(Threads.Atomic{Int}(0), Ref(false), Task[])
    for w in 2:length(plan.xg)
        push!(rt.tasks, Threads.@spawn dagteam_worker!($plan, $rt, $w))
    end
    return rt
end

function dagteam_stop_team!(rt::DagTeamRuntime)
    rt.stop[] = true
    foreach(wait, rt.tasks)
    return nothing
end

# one sweep: reset counters FIRST (parks any laggard on empty queues), rebuild
# queues, wake the team, drain as the coordinator, then the boundary reduction
#
# `diagnostics` (optional Dict) accumulates coordinator-side coarse timers:
# :dagteam_wait_ns (post-drain laggard wait) and :dagteam_reduce_ns (serial
# boundary reduction). Both are SUBSETS of solve!'s :nonself_product_ns —
# existing keys keep their meaning. Only the coordinator writes the dict.
function dagteam_sweep!(plan::DagTeamPlan, rt::DagTeamRuntime, diagnostics=nothing)
    plan.ndone[] = 0
    copyto!(plan.indeg, plan.indeg0)
    lock(plan.qlock)
    empty!(plan.readyQ)
    append!(plan.readyQ, plan.roots)
    empty!(plan.backQ)
    plan.qhint[] = length(plan.readyQ)
    unlock(plan.qlock)
    Threads.atomic_add!(rt.epoch, 1)
    dagteam_drain!(plan, 1)
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    while plan.ndone[] < plan.ntasks   # laggard finishing its last task body
        GC.safepoint()
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    diagnostics === nothing || (diagnostics[:dagteam_wait_ns] += time_ns() - t_stage)
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    dagteam_reduce_u!(plan)
    diagnostics === nothing || (diagnostics[:dagteam_reduce_ns] += time_ns() - t_stage)
    return nothing
end

#------- solve! integration -------#

# init replacement for update_nonself_influence!: capture the external rhs,
# prime u^0 = U x^0 (warm starts), compute L x^0, and subtract both from the
# right-hand side — same result as the lex init's `+= old(=0), -= new` pass
function dagteam_initialize!(right_hand_side, plan::DagTeamPlan{TM,TS,TF},
        strengths::Vector{TF}) where {TM,TS,TF}

    copyto!(plan.external_rhs, right_hand_side)
    @inbounds for k in eachindex(plan.x)
        plan.x[k] = TS(strengths[k])
    end
    n_leaves = length(plan.preds)

    Threads.@threads :static for j in 1:n_leaves
        plan.mup[j] > 0 && dagteam_do_back_product!(plan, j)
    end
    dagteam_reduce_u!(plan)

    # strided worker partition (explicit scratch index — threadid() may
    # exceed nthreads() under Julia's interactive threadpool); lsum rows
    # are leaf-disjoint
    nw = length(plan.xg)
    Threads.@threads :static for w in 1:nw
        for i in w:nw:n_leaves
            base = plan.offset[i]
            n = plan.offset[i + 1] - base
            yv = dagteam_pull!(plan, w, i)
            @inbounds for k in 1:n
                plan.lsum[base + k] = yv[k]
            end
        end
    end

    @inbounds for k in eachindex(right_hand_side)
        right_hand_side[k] -= TF(plan.lsum[k]) + TF(plan.u[k])
    end
    return nothing
end

# the outer iteration's nearfield block: build b = external + farfield, run
# the inner sweeps on the (possibly reduced-precision) sweep state, then write
# the iterate back and rebuild the production rhs invariant
# rhs = external + farfield − Lx − Ux at the new iterate
function dagteam_inner_sweeps!(right_hand_side, extra_right_hand_side,
        strengths::Vector{TF}, plan::DagTeamPlan{TM,TS,TF},
        n_sweeps::Int, diagnostics=nothing) where {TM,TS,TF}

    @inbounds for k in eachindex(plan.b)
        plan.b[k] = TS(plan.external_rhs[k] + extra_right_hand_side[k])
        plan.x[k] = TS(strengths[k])
    end
    # per-worker aggregates: reset per inner-sweep block, folded into the
    # diagnostics dict below (accumulating across the solve's outer iterations
    # like every other key; busy_max/min therefore sum per-block extrema)
    plan.collect_stats[] = diagnostics !== nothing
    if plan.collect_stats[]
        foreach(reset!, plan.stats)
    end
    # team lifecycle timers (:dagteam_spawn_ns / :dagteam_join_ns) are subsets
    # of solve!'s :nonself_product_ns, like the sweep timers above
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    rt = dagteam_start_team!(plan)
    diagnostics === nothing || (diagnostics[:dagteam_spawn_ns] += time_ns() - t_stage)
    try
        for _ in 1:n_sweeps
            dagteam_sweep!(plan, rt, diagnostics)
        end
    finally
        t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
        dagteam_stop_team!(rt)
        diagnostics === nothing || (diagnostics[:dagteam_join_ns] += time_ns() - t_stage)
    end
    if diagnostics !== nothing
        # per-worker aggregate rollup (021 Stage 2). Sums are over the team;
        # busy extrema accumulate per inner-sweep block (imbalance indicator,
        # not a single-block extremum). All are subsets of :nonself_product_ns
        # except the counts.
        busy = [st.busy_lower_ns + st.busy_back_ns for st in plan.stats]
        diagnostics[:dagteam_busy_lower_ns] += sum(st.busy_lower_ns for st in plan.stats)
        diagnostics[:dagteam_busy_back_ns] += sum(st.busy_back_ns for st in plan.stats)
        diagnostics[:dagteam_lockmgmt_ns] += sum(st.lockmgmt_ns for st in plan.stats)
        diagnostics[:dagteam_idle_ns] += sum(st.idle_ns for st in plan.stats)
        diagnostics[:dagteam_busy_max_ns] += maximum(busy)
        diagnostics[:dagteam_busy_min_ns] += minimum(busy)
        diagnostics[:dagteam_empty_pops] += UInt64(sum(st.empty_pops for st in plan.stats))
        diagnostics[:dagteam_n_lower] += UInt64(sum(st.n_lower for st in plan.stats))
        diagnostics[:dagteam_n_back] += UInt64(sum(st.n_back for st in plan.stats))
        diagnostics[:dagteam_team_size] = UInt64(length(plan.stats))
        plan.collect_stats[] = false
    end
    @inbounds for k in eachindex(strengths)
        strengths[k] = TF(plan.x[k])
        right_hand_side[k] = plan.external_rhs[k] + extra_right_hand_side[k] -
                             TF(plan.lsum[k]) - TF(plan.u[k])
    end
    return nothing
end
