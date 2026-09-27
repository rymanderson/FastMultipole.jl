#=##############################################################################
FastGaussSeidel sweep_order=:dagedge — edge-level partial pulls on a static
per-worker schedule (BRAINSTORM 021 L-shortening item #1; design in
FLOWPanel BRAINSTORM/021/fgs_dagedge_design_20260924.md, sized by
fgs_lshortening_gate0_20260924.md: node→edge split shortens the byte-weighted
critical path 289.5 → 68–69 MB/sweep, a 4.25× sweep bound flat in j).

Semantics (iterate-preserving vs :dagteam/:lexicographic — mathematically
identical, NOT bitwise; same certification, no tolerance recalibration):
instead of leaf i waiting for ALL lower predecessors and streaming one
aggregated GEMV, each big lower edge (i,j) (block bytes ≥ θ) is an
independent task computing the partial y_ij = L_ij x_j the moment source j
publishes; small edges (< θ) of a leaf stay aggregated in one task that
accumulates per-edge GEMVs in fixed ascending-source order. When all of leaf
i's partial slots have arrived (per-leaf atomic arrival counter), the worker
that delivered the LAST slot finalizes inline: reduce the slots in fixed
ascending slot order (big edges ascending j, then the small aggregate),
x_i = b_i − (Lx)_i − u_i, cached-LU solve in place, publish. The upper/filler
side (backward products, serial boundary reduction into the frozen u) and
the outer-iteration RHS invariant are reused from :dagteam verbatim.

Scheduler: NO shared ready queue. At plan build a deterministic event-driven
greedy list-scheduling simulation over the edge DAG (exact byte costs;
priorities = byte-weighted downstream critical path; back products are
priority-0 filler) assigns every task to one worker; each worker walks its
own static list in simulated-start order, waiting on each task's dependency
predicate (`published[j] ≥ sweep`) with the plan's :spin/:backoff policy.
Deadlock-free: lists are ordered by simulated start time and the sim only
starts a task after its dependency's simulated completion, so if all workers
were blocked, the blocked head with minimal simulated start would have an
unfinished ancestor list task with strictly smaller simulated start — an
earlier-or-equal blocked head with smaller start, a contradiction. The sim
asserts dependency-completion ≤ start for every task at build.

Determinism: each slot is written by exactly one task, the reduce runs after
all arrivals in fixed slot order, every GEMV is single-threaded with fixed
per-row order, and x_j reads are gated by sequentially consistent published
flags — so values are independent of scheduling: bitwise-identical across
repeats, worker counts, thread counts, and idle policies at fixed θ.
Changing θ regroups the partial sums (mathematically identical, new bits).
=###############################################################################

#------- construction -------#

"""
    build_dagedge_plan(precision, nonself_matrices, sorted_list, index_map,
                       source_tree, target_tree, strengths_by_leaf,
                       targets_by_branch, self_matrices, leaf_lu_cache;
                       nworkers, idle_policy, theta)

Build the `DagEdgePlan` for `sweep_order=:dagedge`: an unmodified
`DagTeamPlan` (same repack, validation, precision modes, LU caches, scratch)
plus the edge-level task partition (θ = `theta` bytes: block bytes
`sizeof(TM)·n_i·n_j ≥ θ` become independent partial tasks, smaller blocks
stay in one per-leaf aggregate task), per-leaf partial slot buffers, and the
static per-worker schedule (see file header). `idle_policy` selects the
dependency-wait pause policy (:spin / :backoff — same semantics as the
:dagteam queue-idle policy; there is no queue here).
"""
function build_dagedge_plan(precision::Symbol, nonself_matrices::Matrices{TF},
        sorted_list::Vector{SVector{2,Int32}}, index_map::Vector{UnitRange{Int}},
        source_tree::Tree, target_tree::Tree,
        strengths_by_leaf::Vector{UnitRange{Int}},
        targets_by_branch::Vector{UnitRange{Int}},
        self_matrices::Matrices{TF}, leaf_lu_cache;
        nworkers::Integer=Threads.nthreads(),
        idle_policy::Symbol=:spin,
        theta::Integer=4096) where TF

    theta >= 0 || throw(ArgumentError(
        "dagedge_theta must be a nonnegative byte count (got $theta)"))
    base = build_dagteam_plan(precision, nonself_matrices, sorted_list,
        index_map, source_tree, target_tree, strengths_by_leaf,
        targets_by_branch, self_matrices, leaf_lu_cache; nworkers, idle_policy)
    return build_dagedge_plan(base, Int(theta))
end

function build_dagedge_plan(base::DagTeamPlan{TM,TS,TF,TLU}, theta::Int) where {TM,TS,TF,TLU}
    n_leaves = length(base.preds)
    offset = base.offset
    nof(i) = offset[i + 1] - offset[i]
    szTM = sizeof(TM)
    szTS = sizeof(TS)

    #--- big/small edge partition over the repacked column blocks ---#

    # column offsets replay the repack's ascending-predecessor layout
    bigs = [Tuple{Int32,Int32}[] for _ in 1:n_leaves]    # (source j, col0)
    smalls = [Tuple{Int32,Int32}[] for _ in 1:n_leaves]
    for i in 1:n_leaves
        c0 = 0
        ni = nof(i)
        for j in base.preds[i]                            # ascending
            nj = nof(j)
            push!((szTM * ni * nj >= theta ? bigs : smalls)[i], (Int32(j), Int32(c0)))
            c0 += nj
        end
        c0 == base.ptot[i] || error("dagedge: column offsets diverged from ptot for leaf $i")
    end
    nslots = [length(bigs[i]) + (isempty(smalls[i]) ? 0 : 1) for i in 1:n_leaves]
    Y = [zeros(TS, nof(i), nslots[i]) for i in 1:n_leaves]

    #--- task costs (bytes) and downstream critical-path priorities ---#

    c_small = [isempty(smalls[i]) ? 0.0 :
               Float64(szTM) * nof(i) * sum(nof(Int(j)) for (j, _) in smalls[i])
               for i in 1:n_leaves]
    c_fin = [Float64(szTS) * (nof(i) * nslots[i] + nof(i)^2) for i in 1:n_leaves]

    # P[i] = finalize cost + longest byte path below publish(i); small edges
    # are charged at their leaf's full aggregate cost (slight overestimate)
    P = zeros(n_leaves)
    for i in n_leaves:-1:1                                # successors have larger index
        m = 0.0
        ni = nof(i)
        for k in base.nsucc[i]
            ce = Float64(szTM) * nof(k) * ni
            ce >= theta || (ce = c_small[k])
            p = ce + P[k]
            p > m && (m = p)
        end
        P[i] = c_fin[i] + m
    end
    sim_edge_L = maximum(P; init=0.0)

    #--- task table ---#

    tasks = DagEdgeTask[]
    prio = Float64[]
    cost = Float64[]
    for i in 1:n_leaves
        ni = nof(i)
        if isempty(base.preds[i])                         # root: finalize is the task
            push!(tasks, DagEdgeTask(0x03, Int32(i), Int32(0), Int32(0), Int32(0)))
            push!(prio, P[i]); push!(cost, c_fin[i])
        end
        for (slot, (j, c0)) in enumerate(bigs[i])
            push!(tasks, DagEdgeTask(0x01, Int32(i), j, Int32(slot), c0))
            push!(prio, Float64(szTM) * ni * nof(Int(j)) + P[i])
            push!(cost, Float64(szTM) * ni * nof(Int(j)))
        end
        if !isempty(smalls[i])
            push!(tasks, DagEdgeTask(0x02, Int32(i), Int32(0), Int32(nslots[i]), Int32(0)))
            push!(prio, c_small[i] + P[i]); push!(cost, c_small[i])
        end
        if base.mup[i] > 0                                # backward product: filler
            push!(tasks, DagEdgeTask(0x04, Int32(i), Int32(0), Int32(0), Int32(0)))
            push!(prio, 0.0); push!(cost, Float64(szTM) * base.mup[i] * ni)
        end
    end

    #--- deterministic greedy list-scheduling simulation ---#

    dep_edge = [Int[] for _ in 1:n_leaves]    # tasks released by publish(j): kinds 1, 4
    dep_small = [Int[] for _ in 1:n_leaves]   # kind-2 tasks counting leaf j
    small_left = zeros(Int, length(tasks))
    small_rel = zeros(Float64, length(tasks))
    for (t, task) in enumerate(tasks)
        if task.kind == 0x01
            push!(dep_edge[Int(task.j)], t)
        elseif task.kind == 0x02
            small_left[t] = length(smalls[Int(task.i)])
            for (j, _) in smalls[Int(task.i)]
                push!(dep_small[Int(j)], t)
            end
        elseif task.kind == 0x04
            push!(dep_edge[Int(task.i)], t)
        end
    end

    nw = length(base.xg)
    lists = [DagEdgeTask[] for _ in 1:nw]
    worker_free = zeros(nw)
    last_start = fill(-1.0, nw)
    start_time = fill(-1.0, length(tasks))
    pub_time = fill(-1.0, n_leaves)
    arrivals = zeros(Int, n_leaves)
    arr_last = zeros(Float64, n_leaves)
    ready = Tuple{Float64,Int}[]              # (-prio, id) min-heap → max-prio pop
    releases = Tuple{Float64,Int}[]           # (time, id) min-heap

    function sim_publish!(i::Int, t::Float64)
        pub_time[i] = t
        for tid in dep_edge[i]
            dagedge_heap_push!(releases, (t, tid))
        end
        for tid in dep_small[i]
            small_rel[tid] = max(small_rel[tid], t)
            (small_left[tid] -= 1) == 0 && dagedge_heap_push!(releases, (small_rel[tid], tid))
        end
        return nothing
    end

    for (t, task) in enumerate(tasks)         # roots are dependency-free
        task.kind == 0x03 && dagedge_heap_push!(releases, (0.0, t))
    end

    nsched = 0
    curt = 0.0
    while nsched < length(tasks)
        while !isempty(releases) && releases[1][1] <= curt
            _, tid = dagedge_heap_pop!(releases)
            dagedge_heap_push!(ready, (-prio[tid], tid))
        end
        if isempty(ready)
            isempty(releases) && error("dagedge schedule: no ready or pending tasks " *
                "with $(length(tasks) - nsched) unscheduled — dependency bug")
            curt = max(curt, releases[1][1])
            continue
        end
        w = argmin(worker_free)
        if worker_free[w] > curt
            curt = worker_free[w]
            continue
        end
        _, tid = dagedge_heap_pop!(ready)
        task = tasks[tid]
        start_time[tid] = curt
        curt >= last_start[w] || error("dagedge schedule: worker list start times not monotone")
        last_start[w] = curt
        push!(lists[w], task)
        fin = curt + cost[tid]
        worker_free[w] = fin
        nsched += 1
        if task.kind == 0x01 || task.kind == 0x02
            i = Int(task.i)
            arrivals[i] += 1
            arr_last[i] = max(arr_last[i], fin)
            if arrivals[i] == nslots[i]        # inline finalize on this worker
                tfin = max(arr_last[i], fin) + c_fin[i]
                worker_free[w] = tfin
                sim_publish!(i, tfin)
            end
        elseif task.kind == 0x03
            sim_publish!(Int(task.i), fin)
        end
    end
    sim_makespan = maximum(worker_free; init=0.0)

    #--- feasibility asserts (mechanical proof obligations) ---#

    tol = 1e-6
    for (tid, task) in enumerate(tasks)
        st = start_time[tid]
        st >= 0 || error("dagedge schedule: task $tid never scheduled")
        if task.kind == 0x01
            pub_time[Int(task.j)] <= st + tol || error(
                "dagedge schedule: edge task ($(task.i),$(task.j)) starts before its source publishes")
        elseif task.kind == 0x02
            for (j, _) in smalls[Int(task.i)]
                pub_time[Int(j)] <= st + tol || error(
                    "dagedge schedule: small-aggregate task of leaf $(task.i) starts before source $j publishes")
            end
        elseif task.kind == 0x04
            pub_time[Int(task.i)] <= st + tol || error(
                "dagedge schedule: back-product task of leaf $(task.i) starts before it publishes")
        end
    end
    all(>=(0.0), pub_time) || error("dagedge schedule: some leaf never publishes")

    n_roots = count(i -> isempty(base.preds[i]), 1:n_leaves)
    ntasks = length(tasks) + (n_leaves - n_roots)   # + one inline finalize per non-root leaf

    return DagEdgePlan{TM,TS,TF,TLU}(base, theta, nslots, smalls, Y, lists,
        [Threads.Atomic{Int}(0) for _ in 1:n_leaves],
        [Threads.Atomic{Int}(0) for _ in 1:n_leaves],
        Threads.Atomic{Int}(0), ntasks, sim_makespan, sim_edge_L)
end

#------- binary min-heap on (key, id) tuples (lexicographic; ties → lowest id) -------#

@inline function dagedge_heap_push!(h::Vector{Tuple{Float64,Int}}, item::Tuple{Float64,Int})
    push!(h, item)
    k = length(h)
    @inbounds while k > 1
        p = k >> 1
        h[k] < h[p] || break
        h[k], h[p] = h[p], h[k]
        k = p
    end
    return h
end

@inline function dagedge_heap_pop!(h::Vector{Tuple{Float64,Int}})
    top = h[1]
    tail = pop!(h)
    n = length(h)
    if n > 0
        h[1] = tail
        k = 1
        @inbounds while true
            l = k << 1
            m = k
            l <= n && h[l] < h[m] && (m = l)
            l + 1 <= n && h[l + 1] < h[m] && (m = l + 1)
            m == k && break
            h[k], h[m] = h[m], h[k]
            k = m
        end
    end
    return top
end

#------- sweep executor -------#

# bounded wait until leaf j has published in sweep s (dependency predicate);
# progress: published is monotone within the block and set by the finalize
# that ends every leaf exactly once per sweep
@inline function dagedge_wait_published(plan::DagEdgePlan, j::Int, s::Int, backoff::Bool)
    if backoff
        delay = 32
        while plan.published[j][] < s
            for _ in 1:delay
                ccall(:jl_cpu_pause, Cvoid, ())
            end
            GC.safepoint()
            delay = min(delay << 1, 4096)
        end
    else
        while plan.published[j][] < s
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
    end
    return nothing
end

# reduce leaf i's partial slots in fixed ascending slot order, form
# x_i = b_i − (Lx)_i − u_i, cached-LU solve in place, publish (SC store
# releases x_i to workers waiting on the flag)
function dagedge_finalize!(plan::DagEdgePlan{TM,TS}, w::Int, i::Int, s::Int) where {TM,TS}
    base = plan.base
    b0 = base.offset[i]
    n = base.offset[i + 1] - b0
    yv = view(base.yb[w], 1:n)
    ns = plan.nslots[i]
    Yi = plan.Y[i]
    if ns == 0
        fill!(yv, zero(TS))
    else
        @inbounds for k in 1:n
            yv[k] = Yi[k, 1]
        end
        @inbounds for c in 2:ns
            @simd for k in 1:n
                yv[k] += Yi[k, c]
            end
        end
    end
    xv = view(base.x, b0 + 1:b0 + n)
    @inbounds for k in 1:n
        base.lsum[b0 + k] = yv[k]                          # saved: Lx at this sweep
        xv[k] = base.b[b0 + k] - yv[k] - base.u[b0 + k]
    end
    ldiv!(base.lus.factorizations[i], xv)
    plan.published[i][] = s
    Threads.atomic_add!(plan.ndone, 1)
    return nothing
end

# slot arrival: the worker delivering the LAST slot finalizes inline
@inline function dagedge_arrive!(plan::DagEdgePlan, w::Int, i::Int, s::Int)
    Threads.atomic_add!(plan.ndone, 1)
    old = Threads.atomic_add!(plan.slotcnt[i], 1)
    old + 1 == plan.nslots[i] && dagedge_finalize!(plan, w, i, s)
    return nothing
end

# big-edge partial y_ij = L_ij x_j into its private slot
function dagedge_do_edge!(plan::DagEdgePlan{TM,TS}, w::Int, task::DagEdgeTask, s::Int) where {TM,TS}
    base = plan.base
    i = Int(task.i)
    j = Int(task.j)
    nj = base.offset[j + 1] - base.offset[j]
    xv = view(base.x, base.offset[j] + 1:base.offset[j + 1])
    A = view(base.Lmat[i], :, Int(task.col0) + 1:Int(task.col0) + nj)
    dagteam_gemv!(view(plan.Y[i], :, Int(task.slot)), A, xv)
    dagedge_arrive!(plan, w, i, s)
    return nothing
end

# small-edge aggregate: per-edge GEMVs accumulated in fixed ascending-source
# order into one slot; sources not yet published are awaited in turn (these
# embedded waits are counted as busy time, not idle — prototype accounting)
function dagedge_do_small!(plan::DagEdgePlan{TM,TS}, w::Int, i::Int, slot::Int,
        s::Int, backoff::Bool) where {TM,TS}
    base = plan.base
    yv = view(plan.Y[i], :, slot)
    first_edge = true
    @inbounds for (j32, c032) in plan.smalls[i]
        j = Int(j32)
        plan.published[j][] < s && dagedge_wait_published(plan, j, s, backoff)
        c0 = Int(c032)
        nj = base.offset[j + 1] - base.offset[j]
        xv = view(base.x, base.offset[j] + 1:base.offset[j + 1])
        A = view(base.Lmat[i], :, c0 + 1:c0 + nj)
        first_edge ? dagteam_gemv!(yv, A, xv) : dagteam_gemv_acc!(yv, A, xv)
        first_edge = false
    end
    dagedge_arrive!(plan, w, i, s)
    return nothing
end

# walk worker w's static list for sweep s (stats mirror the :dagteam drain:
# busy_lower = edge/small/finalize bodies, busy_back = back products,
# idle = pre-task dependency waits, lockmgmt stays 0 — there are no locks)
function dagedge_execute_list!(plan::DagEdgePlan, w::Int, s::Int)
    base = plan.base
    backoff = base.idle_policy === :backoff
    cs = base.collect_stats[]
    st = base.stats[w]
    for task in plan.lists[w]
        k = task.kind
        if k == 0x01 || k == 0x04
            j = k == 0x01 ? Int(task.j) : Int(task.i)
            if plan.published[j][] < s
                t0 = cs ? time_ns() : UInt64(0)
                dagedge_wait_published(plan, j, s, backoff)
                cs && (st.idle_ns += time_ns() - t0; st.empty_pops += 1)
            end
        end
        t1 = cs ? time_ns() : UInt64(0)
        if k == 0x01
            dagedge_do_edge!(plan, w, task, s)
            cs && (st.busy_lower_ns += time_ns() - t1; st.n_lower += 1)
        elseif k == 0x02
            dagedge_do_small!(plan, w, Int(task.i), Int(task.slot), s, backoff)
            cs && (st.busy_lower_ns += time_ns() - t1; st.n_lower += 1)
        elseif k == 0x03
            dagedge_finalize!(plan, w, Int(task.i), s)
            cs && (st.busy_lower_ns += time_ns() - t1; st.n_lower += 1)
        else
            dagteam_do_back_product!(base, Int(task.i))
            Threads.atomic_add!(plan.ndone, 1)
            cs && (st.busy_back_ns += time_ns() - t1; st.n_back += 1)
        end
    end
    return nothing
end

#------- worker team lifecycle (epoch protocol shared with :dagteam) -------#

function dagedge_worker!(plan::DagEdgePlan, rt::DagTeamRuntime, w::Int)
    my_epoch = 0
    while true
        while rt.epoch[] == my_epoch
            rt.stop[] && return nothing
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
        my_epoch += 1
        dagedge_execute_list!(plan, w, my_epoch)
    end
end

function dagedge_start_team!(plan::DagEdgePlan)
    rt = DagTeamRuntime(Threads.Atomic{Int}(0), Ref(false), Task[], DagCoopTeam[])
    for w in 2:length(plan.lists)
        push!(rt.tasks, Threads.@spawn dagedge_worker!($plan, $rt, $w))
    end
    return rt
end

# one sweep: reset arrival counters + completion counter (workers are parked
# at the epoch barrier — the previous sweep fully drained), wake the team,
# walk the coordinator's list, wait out laggards, boundary-reduce. Published
# flags are NOT reset (monotone in the sweep number within a block).
function dagedge_sweep!(plan::DagEdgePlan, rt::DagTeamRuntime, s::Int, diagnostics=nothing)
    plan.ndone[] = 0
    @inbounds for c in plan.slotcnt
        c[] = 0
    end
    Threads.atomic_add!(rt.epoch, 1)
    dagedge_execute_list!(plan, 1, s)
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    while plan.ndone[] < plan.ntasks
        GC.safepoint()
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    diagnostics === nothing || (diagnostics[:dagteam_wait_ns] += time_ns() - t_stage)
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    dagteam_reduce_u!(plan.base)
    diagnostics === nothing || (diagnostics[:dagteam_reduce_ns] += time_ns() - t_stage)
    return nothing
end

#------- solve! integration -------#

# init is layout-independent: reuse the :dagteam pass on the wrapped plan
dagteam_initialize!(right_hand_side, plan::DagEdgePlan, strengths) =
    dagteam_initialize!(right_hand_side, plan.base, strengths)

# mirror of dagteam_inner_sweeps! (same state loads, RHS invariant, and
# :dagteam_* diagnostics keys — lockmgmt is identically zero here)
function dagedge_inner_sweeps!(right_hand_side, extra_right_hand_side,
        strengths::Vector{TF}, plan::DagEdgePlan{TM,TS,TF},
        n_sweeps::Int, diagnostics=nothing) where {TM,TS,TF}

    base = plan.base
    @inbounds for k in eachindex(base.b)
        base.b[k] = TS(base.external_rhs[k] + extra_right_hand_side[k])
        base.x[k] = TS(strengths[k])
    end
    @inbounds for p in plan.published
        p[] = 0
    end
    base.collect_stats[] = diagnostics !== nothing
    if base.collect_stats[]
        foreach(reset!, base.stats)
    end
    t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
    rt = dagedge_start_team!(plan)
    diagnostics === nothing || (diagnostics[:dagteam_spawn_ns] += time_ns() - t_stage)
    try
        for s in 1:n_sweeps
            dagedge_sweep!(plan, rt, s, diagnostics)
        end
    finally
        t_stage = diagnostics === nothing ? UInt64(0) : time_ns()
        dagteam_stop_team!(rt)
        diagnostics === nothing || (diagnostics[:dagteam_join_ns] += time_ns() - t_stage)
    end
    if diagnostics !== nothing
        busy = [st.busy_lower_ns + st.busy_back_ns for st in base.stats]
        diagnostics[:dagteam_busy_lower_ns] += sum(st.busy_lower_ns for st in base.stats)
        diagnostics[:dagteam_busy_back_ns] += sum(st.busy_back_ns for st in base.stats)
        diagnostics[:dagteam_lockmgmt_ns] += sum(st.lockmgmt_ns for st in base.stats)
        diagnostics[:dagteam_idle_ns] += sum(st.idle_ns for st in base.stats)
        diagnostics[:dagteam_busy_max_ns] += maximum(busy)
        diagnostics[:dagteam_busy_min_ns] += minimum(busy)
        diagnostics[:dagteam_empty_pops] += UInt64(sum(st.empty_pops for st in base.stats))
        diagnostics[:dagteam_n_lower] += UInt64(sum(st.n_lower for st in base.stats))
        diagnostics[:dagteam_n_back] += UInt64(sum(st.n_back for st in base.stats))
        diagnostics[:dagteam_team_size] = UInt64(length(base.stats))
        base.collect_stats[] = false
    end
    @inbounds for k in eachindex(strengths)
        strengths[k] = TF(base.x[k])
        right_hand_side[k] = base.external_rhs[k] + extra_right_hand_side[k] -
                             TF(base.lsum[k]) - TF(base.u[k])
    end
    return nothing
end
