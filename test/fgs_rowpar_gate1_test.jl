#=##############################################################################
FGS source-major row-parallel schedule — gate-1 correctness model
(FLOWPanel BRAINSTORM 021, fgs_acceleration_recommendation_20260918.md §gates).

A *shadow executor* implements the candidate schedule against the production
FastGaussSeidel state, without touching solve.jl: the lexicographic leaf loop
is kept; per leaf, the coordinator solves the diagonal block, then the nonself
product AND its scatter are partitioned into disjoint contiguous row tiles
executed in parallel. Tiles preserve the production RHS semantics per row:
old is saved, the new product overwrites the same buffer, and each owned rhs
row receives `+= old` then `-= new` as two operations (never a delta).

Gate-1 claims verified here, in Float64 only (precision comes later):
 1. ntiles=1 shadow sweep is BITWISE identical to production :lexicographic
    gs_sweep! (proves the shadow executor reproduces the reference exactly);
 2. multi-tile parallel sweeps match lex on strengths / products / RHS /
    old-influence / residual to tight tolerance across multiple sweeps,
    including nonzero starts (sweeps 2,3 continue from sweep-1 state);
    bitwise status is reported (BLAS sub-range GEMV is expected but not
    guaranteed to be bit-identical — recommendation §traps);
 3. holds on a rigidly transformed solver (transform_solver! fixture);
 4. holds with multiple source systems when the constructor supports the
    fixture (branch-spanning target rows exercise the row→target map).
=###############################################################################

if !(@isdefined FastMultipole)   # standalone execution support
    using FastMultipole
    using FastMultipole.LinearAlgebra
    using FastMultipole.StaticArrays
    using Random, Test
    include("gravitational.jl")

    # solver-compat overloads (duplicated from solve_test.jl, which owns them
    # inside runtests.jl; standalone runs need them here)
    function FastMultipole.influence!(influence, target_buffer, derivatives_switch::FastMultipole.DerivativesSwitch, ::Gravitational, source_buffer)
        influence .= view(target_buffer, FastMultipole.scalar_potential_index(derivatives_switch), :)
    end
    function FastMultipole.target_influence_to_buffer!(target_buffer, i_buffer, derivatives_switch::FastMultipole.DerivativesSwitch{PS}, target_system::Gravitational, i_target) where PS
        PS && (target_buffer[FastMultipole.scalar_potential_index(derivatives_switch), i_buffer] = target_system.potential[1, i_target])
    end
    function FastMultipole.value_to_strength!(source_buffer, ::Gravitational, i_body, value)
        source_buffer[5, i_body] = value
    end
    function FastMultipole.strength_to_value(strength, ::Gravitational)
        return strength[1]
    end
    function FastMultipole.buffer_to_system_strength!(system::Gravitational, i_body, source_buffer, i_buffer)
        position = system.bodies[i_body].position
        radius = system.bodies[i_body].radius
        strength = source_buffer[5, i_buffer]
        system.bodies[i_body] = eltype(system.bodies)(position, radius, strength)
    end
end

#------- shadow executor -------#

# Row-tiled version of update_nonself_influence! for one source leaf: tiles
# are contiguous local row ranges of the leaf's tall matrix; each tile saves
# old, computes new (column-major streaming GEMV on the row sub-range), and
# scatters to its owned rhs rows. Distinct direct-list targets of one source
# occupy disjoint rhs ranges, and tiles split local rows disjointly, so all
# writes are tile-disjoint.
function rowpar_update_nonself!(right_hand_side, strengths, nonself_matrices,
        old_influence_storage, i_leaf::Int, target_tree, strengths_by_leaf,
        index_map, direct_list, targets_by_branch; ntiles::Int)

    mat, target_influence = FastMultipole.get_matrix_vector(nonself_matrices, i_leaf)
    m = size(mat, 1)
    m == 0 && return nothing
    rhs_offset = nonself_matrices.rhs_offsets[i_leaf]
    old_influence = view(old_influence_storage, rhs_offset:rhs_offset + m - 1)
    leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

    # local segment map (ascending direct-list order, same as production scatter)
    seg_local_start = Int[]; seg_global_start = Int[]; seg_len = Int[]
    i_start = 1
    for index in index_map[i_leaf]
        i_target, _ = direct_list[index]
        n_t = FastMultipole.get_n_bodies(target_tree.branches[i_target].bodies_index)
        push!(seg_local_start, i_start)
        push!(seg_global_start, first(targets_by_branch[i_target]))
        push!(seg_len, n_t)
        i_start += n_t
    end
    @assert i_start - 1 == m

    tiles = [div((w-1)*m, ntiles)+1 : div(w*m, ntiles) for w in 1:ntiles]
    Threads.@threads for w in 1:ntiles
        r = tiles[w]
        isempty(r) && continue
        # save old, compute new on this tile's rows (same buffers as production)
        @views old_influence[r] .= target_influence[r]
        mul!(view(target_influence, r), view(mat, r, :), leaf_strengths)
        # scatter owned rows: `+= old` then `-= new`, per row — identical
        # elementwise arithmetic to the production segment broadcasts
        k = searchsortedlast(seg_local_start, first(r))
        row = first(r)
        while row <= last(r)
            hi = min(seg_local_start[k] + seg_len[k] - 1, last(r))
            g = seg_global_start[k] + (row - seg_local_start[k])
            @inbounds for t in 0:(hi - row)
                right_hand_side[g+t] += old_influence[row+t]
                right_hand_side[g+t] -= target_influence[row+t]
            end
            row = hi + 1
            k += 1
        end
    end
    return nothing
end

# full candidate sweep: production leaf solves, row-tiled nonself updates
function rowpar_sweep!(fgs; ntiles::Int)
    for (i_leaf, _) in enumerate(fgs.source_tree.leaf_index)
        leaf_strengths = view(fgs.strengths, fgs.strengths_by_leaf[i_leaf])
        FastMultipole.solve_leaf!(leaf_strengths, fgs.self_matrices,
            fgs.leaf_lu_cache, i_leaf)
        if length(fgs.direct_list) > 0
            rowpar_update_nonself!(fgs.self_matrices.rhs, fgs.strengths,
                fgs.nonself_matrices, fgs.old_influence_storage, i_leaf,
                fgs.target_tree, fgs.strengths_by_leaf, fgs.index_map,
                fgs.direct_list, fgs.targets_by_branch; ntiles)
        end
    end
    return nothing
end

function lex_sweep!(fgs)
    FastMultipole.gs_sweep!(fgs.strengths, fgs.self_matrices, fgs.leaf_lu_cache,
        fgs.self_matrices.rhs, fgs.nonself_matrices, fgs.old_influence_storage,
        fgs.source_tree, fgs.target_tree, fgs.strengths_by_leaf, fgs.index_map,
        fgs.direct_list, fgs.targets_by_branch, fgs, false)
    return nothing
end

# state comparison: (bitwise?, worst relative deviation over the five fields)
function compare_state(fgs_a, fgs_b)
    fields = ((:strengths, f -> f.strengths),
              (:products,  f -> f.nonself_matrices.rhs),
              (:rhs,       f -> f.self_matrices.rhs),
              (:old,       f -> f.old_influence_storage))
    bitwise = true
    worst = 0.0
    for (_, get) in fields
        a, b = get(fgs_a), get(fgs_b)
        bitwise &= a == b
        scale = max(norm(a, Inf), norm(b, Inf), eps())
        worst = max(worst, norm(a .- b, Inf) / scale)
    end
    res_a = FastMultipole.residual!(fgs_a.residual_vector, fgs_a.self_matrices,
        fgs_a.strengths, fgs_a.strengths_by_leaf)
    res_b = FastMultipole.residual!(fgs_b.residual_vector, fgs_b.self_matrices,
        fgs_b.strengths, fgs_b.strengths_by_leaf)
    bitwise &= res_a == res_b
    worst = max(worst, abs(res_a - res_b) / max(abs(res_a), abs(res_b), eps()))
    return bitwise, worst
end

function seed_state!(fgs)
    fgs.self_matrices.rhs .= sin.(eachindex(fgs.self_matrices.rhs))
    fgs.nonself_matrices.rhs .= 0
    fgs.old_influence_storage .= 0
    fgs.strengths .= 0
    return nothing
end

# run `nsweeps` on both objects, comparing after every sweep
function compare_schedules(make_ref, make_cand; ntiles, nsweeps=3, label="")
    fgs_ref, fgs_cand = make_ref(), make_cand()
    seed_state!(fgs_ref); seed_state!(fgs_cand)
    all_bitwise = true
    worst = 0.0
    for s in 1:nsweeps
        lex_sweep!(fgs_ref)
        rowpar_sweep!(fgs_cand; ntiles)
        bw, w = compare_state(fgs_ref, fgs_cand)
        all_bitwise &= bw
        worst = max(worst, w)
    end
    @test worst <= 1e-13
    @test any(!iszero, fgs_ref.strengths)   # non-vacuous
    return all_bitwise, worst
end

#------- fixtures -------#

@testset "FGS row-parallel schedule: gate-1 correctness model" begin

Random.seed!(20260918)

system = generate_gravitational(20260918, 800)
FastMultipole.direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

make_fgs() = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false, sweep_order=:lexicographic)

fgs_probe = make_fgs()
@test length(fgs_probe.source_tree.leaf_index) > 1
@test !isempty(fgs_probe.direct_list)

# --- 1. serial shadow executor is bitwise-identical to production lex -------
bw1, worst1 = compare_schedules(make_fgs, make_fgs; ntiles=1)
@test bw1     # ntiles=1 must be exact: same ops, same order, same buffers

# --- 2. parallel tiles (uneven and even splits), multiple sweeps ------------
for ntiles in (2, 3, min(4, Threads.nthreads()))
    bw, worst = compare_schedules(make_fgs, make_fgs; ntiles)
    @info "gate-1 single-system" ntiles nthreads=Threads.nthreads() bitwise=bw worst_rel_dev=worst
end

# --- 3. rigidly transformed solver ------------------------------------------
θ = 0.7; R = SMatrix{3,3}([cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1.0])
t = SVector(0.1, -0.2, 0.3)
function make_transformed()
    sys = generate_gravitational(20260918, 800)
    fgs = FastMultipole.FastGaussSeidel((sys,), (sys,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:lexicographic)
    for i in eachindex(sys.bodies)
        b = sys.bodies[i]
        sys.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    FastMultipole.transform_solver!(fgs, (sys,), R, t)
    return fgs
end
bw3, worst3 = compare_schedules(make_transformed, make_transformed; ntiles=min(4, Threads.nthreads()))
@info "gate-1 transformed" bitwise=bw3 worst_rel_dev=worst3

# --- 4. multiple source systems (branch-spanning target rows) ---------------
sysA = generate_gravitational(11, 300)
sysB = generate_gravitational(22, 300)
multi_supported = try
    FastMultipole.FastGaussSeidel((sysA, sysB), (sysA, sysB);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:lexicographic)
    true
catch err
    @info "multi-system FGS fixture unsupported here; implementation-side gate must cover it" err=sprint(showerror, err)
    false
end
if multi_supported
    make_multi() = FastMultipole.FastGaussSeidel((sysA, sysB), (sysA, sysB);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:lexicographic)
    bw4, worst4 = compare_schedules(make_multi, make_multi; ntiles=min(4, Threads.nthreads()))
    @info "gate-1 multi-system" bitwise=bw4 worst_rel_dev=worst4
end

end
