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

# same-precision: BLAS (AbstractMatrix admits the contiguous column-block
# views used by the :dagedge edge tasks)
@inline dagteam_gemv!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}) where T = mul!(y, A, x)

# Float32 storage, Float64 state (:f32conv): convert on load, column-major
function dagteam_gemv!(y::AbstractVector{Float64}, A::AbstractMatrix{Float32}, x::AbstractVector{Float64})
    fill!(y, 0.0)
    @inbounds for jc in axes(A, 2)
        s = x[jc]
        @simd for k in eachindex(y)
            y[k] = muladd(Float64(A[k, jc]), s, y[k])
        end
    end
    return y
end

# accumulating variants (y += A x), used by the :dagedge small-edge aggregate
@inline dagteam_gemv_acc!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}) where T = mul!(y, A, x, true, true)

function dagteam_gemv_acc!(y::AbstractVector{Float64}, A::AbstractMatrix{Float32}, x::AbstractVector{Float64})
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

`coop` (033 A-R2 prototype) sets the cooperative team width `teamw_cap` and
the initial `teamw[]` (see `DagTeamPlan`): 1 (default) is the solo production
executor, bit-identical to the pre-coop behavior; widths 2 and 4 are the
measured A-R1 operating points (any 1 ≤ coop ≤ nworkers is accepted).

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
        idle_policy::Symbol=:spin,
        coop::Integer=1,
        setup_diagnostics=nothing) where TF
    # `setup_diagnostics` (optional Dict) records one-shot ctor-side stage
    # timers (BRAINSTORM 033 B-I2); same idiom as the solve-time `diagnostics`

    precision in (:f64, :f32conv, :f32full) || throw(ArgumentError(
        "dagteam_precision must be :f64, :f32conv, or :f32full (got $(repr(precision)))"))
    idle_policy in (:spin, :backoff) || throw(ArgumentError(
        "dagteam idle_policy must be :spin or :backoff (got $(repr(idle_policy)))"))
    TM = precision === :f64 ? TF : Float32
    TS = precision === :f32full ? Float32 : TF

    n_leaves = length(source_tree.leaf_index)

    t_stage = setup_diagnostics === nothing ? UInt64(0) : time_ns()

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

    if setup_diagnostics !== nothing
        setup_diagnostics[:dagplan_edges_ns] = time_ns() - t_stage
        t_stage = time_ns()
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

    if setup_diagnostics !== nothing
        setup_diagnostics[:dagplan_alloc_ns] = time_ns() - t_stage
        t_stage = time_ns()
    end

    # repack: tall-matrix rows follow index_map segment order; within a
    # segment, rows follow the branch's ascending global rows, i.e. whole
    # leaves lo:hi in ascending order (asserted above)
    for j in 1:n_leaves
        # a source leaf with no direct blocks has nothing to repack; with an
        # entirely empty direct list, nonself_matrices is EmptyMatrices (no
        # per-leaf entries at all), so indexing it here would throw
        isempty(index_map[j]) && continue
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

    if setup_diagnostics !== nothing
        setup_diagnostics[:dagplan_repack_ns] = time_ns() - t_stage
        t_stage = time_ns()
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

    if setup_diagnostics !== nothing
        setup_diagnostics[:dagplan_prio_ns] = time_ns() - t_stage
        t_stage = time_ns()
    end

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

    # cooperative-team scratch (033 A-R2): capacity for any width 2..teamw_cap
    # — at width w there are fld(nw, w) teams, maximized at w=2. Solo builds
    # (coop=1, the default) allocate nothing.
    coop >= 1 || throw(ArgumentError("dagteam coop must be >= 1 (got $coop)"))
    teamw_cap = clamp(Int(coop), 1, nw)
    if teamw_cap > 1
        max_mup = max(1, maximum(mup; init=1))
        xgt = [zeros(TS, max_ptot) for _ in 1:fld(nw, 2)]
        qb = [zeros(TS, max_mup) for _ in 1:nw]
    else
        xgt = Vector{TS}[]
        qb = Vector{TS}[]
    end

    if setup_diagnostics !== nothing
        setup_diagnostics[:dagplan_scratch_ns] = time_ns() - t_stage
        t_stage = time_ns()
    end

    if TS === TF
        leaf_lu_cache isa LeafLUCache || throw(ArgumentError(
            "sweep_order=:dagteam requires cache_leaf_lu=true"))
        lus = leaf_lu_cache
    else
        lus = build_leaf_lu_cache_as(TS, self_matrices)
    end

    setup_diagnostics === nothing ||
        (setup_diagnostics[:dagplan_lu_ns] = time_ns() - t_stage)

    indeg = copy(indeg0)
    readyQ = Int[]; sizehint!(readyQ, n_leaves)
    backQ = Int[]; sizehint!(backQ, n_leaves)
    ntasks = n_leaves + count(>(0), mup)

    return DagTeamPlan{TM,TS,TF,typeof(lus)}(preds, nsucc, uppers, prio, indeg0,
        roots, ptot, mup, Lmat, Umat, red, offset, x, b, u, lsum, q,
        external_rhs, lus, xg, yb,
        Ref(teamw_cap), teamw_cap, xgt, qb,
        indeg, readyQ, backQ, Threads.SpinLock(),
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

# post-pull tail shared bit-identically by the solo and cooperative lower
# paths: save Lx, form b − y − u, solve the cached LU in place, publish
# successors under the queue lock, and count completion
@inline function dagteam_finish_lower!(plan::DagTeamPlan{TM,TS}, i::Int,
        yv::AbstractVector{TS}) where {TM,TS}
    base = plan.offset[i]
    n = plan.offset[i + 1] - base
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

function dagteam_do_lower!(plan::DagTeamPlan{TM,TS}, w::Int, i::Int) where {TM,TS}
    yv = dagteam_pull!(plan, w, i)
    dagteam_finish_lower!(plan, i, yv)
    return nothing
end

# backward product q_j = U_j x_j at the source's current (post-solve) strengths
@inline function dagteam_do_back_product!(plan::DagTeamPlan, j::Int)
    xv = view(plan.x, plan.offset[j] + 1:plan.offset[j + 1])
    dagteam_gemv!(plan.q[j], plan.Umat[j], xv)
    return nothing
end

#------- cooperative leaf products (BRAINSTORM 033 A-R2 prototype) -------#
#
# When plan.teamw[] > 1, workers are grouped into static teams of tw: the
# team's PUBLISHER runs the ordinary drain loop (pops tasks, does the b−y−u
# update, cached LU solve and publication alone), and its tw−1 TEAMMATES are
# dedicated helpers that never touch the queues. A popped GEMV task (lower
# pull or backward product) is executed cooperatively in the A-R1-selected
# COLUMN layout: contiguous column blocks (fixed even partition, member m of
# 0..tw−1 owns block m) into private per-member partial vectors, then a fixed
# ascending-member-order reduction by the publisher — deterministic for fixed
# width/partition, exact-arithmetic equivalent to the solo GEMV (A-T1 column
# clause; NOT bitwise). Publisher/teammate handoff is a seq/arrived atomic
# protocol; all spin waits are GC-safepointed with a rare bounded-yield escape
# (A-R1 harness lesson: required to avoid GC/scheduler deadlock).
#
# Memory ordering: the publisher popped the task under plan.qlock, whose
# acquire pairs with the lock release that published every predecessor's
# strengths, so the publisher sees the finalized plan.x; the publisher's
# seq_cst store of job/seq followed by the teammate's seq_cst load of seq
# extends that happens-before edge to the teammates. Teammate partial writes
# happen-before their atomic arrived increment, which the publisher's
# acquire-read of arrived pairs with before reducing.

struct DagCoopTeam
    tw::Int                       # team width (workers per team)
    t::Int                        # team index (selects plan.xgt[t])
    pub::Int                      # publisher's worker index; member m uses scratch pub + m
    job::Threads.Atomic{Int}      # +i = lower pull of leaf i; −j = backward product of source j
    seq::Threads.Atomic{Int}      # job epoch; teammates run one job per increment
    arrived::Threads.Atomic{Int}  # teammate completions for the current job
    stop::Threads.Atomic{Bool}
end
DagCoopTeam(tw::Int, t::Int, pub::Int) = DagCoopTeam(tw, t, pub,
    Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
    Threads.Atomic{Bool}(false))

# split selection (A-T3 practical rule): A-R1 measured every real R4 leaf
# shape down to p25 clearing the split threshold at w=2 and w=4 (effective
# h_w ≈ 0.4–3 µs vs products ≥ 10 µs), so the prototype splits essentially
# every product — the only guard is a degeneracy floor of two columns per
# member so no member ever owns an empty block. Deliberately NOT restricted
# to hot leaves (A-T3 binding note).
@inline dagteam_coop_eligible(ncols::Int, tw::Int) = ncols >= 2 * tw

# member m's contiguous column block of 1:ncols (fixed even partition —
# deterministic for fixed width)
@inline function dagteam_coop_cols(ncols::Int, tw::Int, m::Int)
    return (div(ncols * m, tw) + 1):div(ncols * (m + 1), tw)
end

# GC-cooperative spin until `at[] >= target` (publisher waiting on arrivals)
@inline function dagteam_coop_wait!(at::Threads.Atomic{Int}, target::Int)
    spins = 0
    while at[] < target
        GC.safepoint()
        ccall(:jl_cpu_pause, Cvoid, ())
        spins += 1
        spins < 10_000 || (yield(); spins = 0)
    end
    return nothing
end

# member m's lower partial for leaf i: gather this member's column slice of
# the ascending-predecessor state into the team's SHARED gather buffer
# (members write disjoint ranges) and compute the private partial
# y^{(m)} = Lmat[i][:, cols] * xgt[cols] into yb[pub + m]
function dagteam_coop_lower_partial!(plan::DagTeamPlan{TM,TS}, xgt::Vector{TS},
        i::Int, w_scratch::Int, m::Int, tw::Int) where {TM,TS}
    cols = dagteam_coop_cols(plan.ptot[i], tw, m)
    off = 0
    @inbounds for j in plan.preds[i]
        nj = plan.offset[j + 1] - plan.offset[j]
        lo = max(off + 1, first(cols))
        hi = min(off + nj, last(cols))
        lo <= hi && copyto!(xgt, lo, plan.x, plan.offset[j] + (lo - off), hi - lo + 1)
        off += nj
        off >= last(cols) && break
    end
    n = plan.offset[i + 1] - plan.offset[i]
    yv = view(plan.yb[w_scratch], 1:n)
    dagteam_gemv!(yv, view(plan.Lmat[i], :, cols), view(xgt, cols))
    return nothing
end

# member m's backward-product partial for source j: column block of the
# source's own strengths; member 0 (publisher) writes q[j] directly, members
# m > 0 write private partials in qb[pub + m]
function dagteam_coop_back_partial!(plan::DagTeamPlan{TM,TS}, j::Int,
        w_scratch::Int, m::Int, tw::Int) where {TM,TS}
    nj = plan.offset[j + 1] - plan.offset[j]
    cols = dagteam_coop_cols(nj, tw, m)
    xv = view(plan.x, plan.offset[j] + first(cols):plan.offset[j] + last(cols))
    A = view(plan.Umat[j], :, cols)
    if m == 0
        dagteam_gemv!(plan.q[j], A, xv)
    else
        dagteam_gemv!(view(plan.qb[w_scratch], 1:plan.mup[j]), A, xv)
    end
    return nothing
end

# publisher side: dispatch the job to the team, compute member 0's own
# partial, wait for the teammates, and reduce in fixed ascending member order
@inline function dagteam_coop_dispatch!(coop::DagCoopTeam, job::Int)
    coop.arrived[] = 0
    coop.job[] = job
    Threads.atomic_add!(coop.seq, 1)
    return nothing
end

function dagteam_coop_do_lower!(plan::DagTeamPlan{TM,TS}, coop::DagCoopTeam,
        i::Int) where {TM,TS}
    tw = coop.tw
    dagteam_coop_dispatch!(coop, i)
    dagteam_coop_lower_partial!(plan, plan.xgt[coop.t], i, coop.pub, 0, tw)
    dagteam_coop_wait!(coop.arrived, tw - 1)
    n = plan.offset[i + 1] - plan.offset[i]
    yv = view(plan.yb[coop.pub], 1:n)
    @inbounds for m in 1:tw - 1                 # fixed ascending block order
        pv = plan.yb[coop.pub + m]
        @simd for k in 1:n
            yv[k] += pv[k]
        end
    end
    dagteam_finish_lower!(plan, i, yv)
    return nothing
end

function dagteam_coop_do_back!(plan::DagTeamPlan{TM,TS}, coop::DagCoopTeam,
        j::Int) where {TM,TS}
    tw = coop.tw
    dagteam_coop_dispatch!(coop, -j)
    dagteam_coop_back_partial!(plan, j, coop.pub, 0, tw)
    dagteam_coop_wait!(coop.arrived, tw - 1)
    qj = plan.q[j]
    mupj = plan.mup[j]
    @inbounds for m in 1:tw - 1                 # fixed ascending block order
        pv = plan.qb[coop.pub + m]
        @simd for k in 1:mupj
            qj[k] += pv[k]
        end
    end
    return nothing
end

# dedicated teammate loop (member m of coop's team, worker index coop.pub+m):
# one job per seq increment. my_seq starts at the constructed value 0; the
# publisher cannot advance seq past a job until every teammate has arrived,
# so no job is ever skipped even if this task is scheduled late (the
# publisher's bounded-yield wait cedes the CPU until it runs).
function dagteam_teammate!(plan::DagTeamPlan, coop::DagCoopTeam, m::Int)
    w_scratch = coop.pub + m
    tw = coop.tw
    my_seq = 0
    while true
        spins = 0
        while coop.seq[] == my_seq
            coop.stop[] && return nothing
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
            spins += 1
            spins < 10_000 || (yield(); spins = 0)
        end
        my_seq += 1
        job = coop.job[]
        if job > 0
            dagteam_coop_lower_partial!(plan, plan.xgt[coop.t], job, w_scratch, m, tw)
        else
            dagteam_coop_back_partial!(plan, -job, w_scratch, m, tw)
        end
        Threads.atomic_add!(coop.arrived, 1)
    end
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
function dagteam_drain!(plan::DagTeamPlan, w::Int,
        coop::Union{Nothing,DagCoopTeam}=nothing)
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
                nj = plan.offset[task + 1] - plan.offset[task]
                if coop !== nothing && dagteam_coop_eligible(nj, coop.tw)
                    dagteam_coop_do_back!(plan, coop, task)
                else
                    dagteam_do_back_product!(plan, task)
                end
                Threads.atomic_add!(plan.ndone, 1)
                cs && (st.busy_back_ns += time_ns() - t1; st.n_back += 1)
            else
                if coop !== nothing && dagteam_coop_eligible(plan.ptot[task], coop.tw)
                    dagteam_coop_do_lower!(plan, coop, task)
                else
                    dagteam_do_lower!(plan, w, task)
                end
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
    coops::Vector{DagCoopTeam}   # empty at teamw[] == 1 (solo, production)
end

function dagteam_worker!(plan::DagTeamPlan, rt::DagTeamRuntime, w::Int,
        coop::Union{Nothing,DagCoopTeam}=nothing)
    my_epoch = 0
    while true
        while rt.epoch[] == my_epoch
            rt.stop[] && return nothing
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
        my_epoch += 1
        dagteam_drain!(plan, w, coop)
    end
end

# teamw[] == 1 (production default): identical to the historical team —
# workers 2..nw all run the solo drain loop. teamw[] = tw > 1: workers are
# grouped into fld(nw, tw) static teams of tw consecutive indices; worker
# (t−1)tw+1 is team t's publisher (worker 1 = coordinator = team 1's
# publisher), the following tw−1 are its dedicated teammates, and any
# leftover workers (nw mod tw) run the solo drain loop. NOTE (A-T1
# determinism clause): with leftovers present, an eligible task computes as
# one GEMV (solo pop) or as a block-split reduction (team pop) depending on
# which worker wins the pop, so run-to-run bitwise reproducibility is only
# guaranteed when nw % tw == 0 (all A-R2-validated configurations);
# exact-arithmetic equivalence and certified accuracy hold either way.
function dagteam_start_team!(plan::DagTeamPlan)
    nw = length(plan.xg)
    tw = plan.teamw[]
    1 <= tw <= plan.teamw_cap || throw(ArgumentError(
        "dagteam teamw[] = $tw outside 1:$(plan.teamw_cap) (teamw_cap fixed at construction — rebuild with a larger coop)"))
    nteams = tw > 1 ? fld(nw, tw) : 0
    coops = [DagCoopTeam(tw, t, (t - 1) * tw + 1) for t in 1:nteams]
    rt = DagTeamRuntime(Threads.Atomic{Int}(0), Ref(false), Task[], coops)
    for w in 2:nw
        t, m = fldmod(w - 1, max(tw, 1))          # team t+1, member m of it
        if tw > 1 && t < nteams && m > 0
            coop = coops[t + 1]
            push!(rt.tasks, Threads.@spawn dagteam_teammate!($plan, $coop, $m))
        else
            coop = (tw > 1 && t < nteams && m == 0) ? coops[t + 1] : nothing
            push!(rt.tasks, Threads.@spawn dagteam_worker!($plan, $rt, $w, $coop))
        end
    end
    return rt
end

function dagteam_stop_team!(rt::DagTeamRuntime)
    rt.stop[] = true
    for coop in rt.coops
        coop.stop[] = true
    end
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
    dagteam_drain!(plan, 1, isempty(rt.coops) ? nothing : rt.coops[1])
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
