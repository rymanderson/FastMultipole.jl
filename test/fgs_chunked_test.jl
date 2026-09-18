#=##############################################################################
FastGaussSeidel sweep_order=:chunked (FLOWPanel BRAINSTORM 021 chunked hybrid
sweep, plan fgs_chunked_hybrid_plan_20260918.md):

1. chunk-map validity + purity — chunks tile 1:n_leaves, contiguous,
   nonempty; scatter partition covers every nonempty segment exactly once
   with the unsplit path's offsets; intra segments write only rows owned by
   the source's own chunk; identical across repeated construction;
2. single-chunk equivalence — a :chunked sweep with nchunks=1 is BITWISE
   identical to the :lexicographic sweep (strengths and self_matrices.rhs);
3. lexicographic regression guard — the :lexicographic sweep stays bitwise
   identical to an inlined replica of the historical serial loop (guards the
   compute/scatter infrastructure against future refactors);
4. threaded-sweep equivalence — a multi-chunk :chunked sweep is BITWISE
   identical to a serial chunk-major reference (parallel GS phase + deferred
   ascending cross-chunk scatter run as plain serial loops); the reference
   never depends on nthreads, so equality at any thread count implies
   thread-count invariance in-process;
5. thread-count invariance across processes — a subprocess at -t 4 (and one
   at -t 1) reproduces the parent's fixed-iteration solve bit-for-bit;
6. repeatability + iteration health — cold fixed-iteration chunked solves are
   bit-identical across repeats and short histories stay finite.

Cross-thread-count bitwise verification at the campaign scale runs in
FLOWPanel's benchmark/fgs_determinism_probe.jl matrix.
=###############################################################################

@testset "Fast Gauss Seidel: chunked sweeps" begin

# premise guards: multiple leaves, nonempty direct list, >1 chunk
system = generate_gravitational(20260918, 800)
direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

make_fgs(sweep_order; chunks=4) = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false, sweep_order, chunks)

fgs = make_fgs(:chunked)
n_leaves = length(fgs.source_tree.leaf_index)
@test n_leaves > 1
@test !isempty(fgs.direct_list)
@test length(fgs.chunk_ranges) == 4          # premise: actual parallelism exists

# --- 1. chunk map validity + purity -----------------------------------------
@test vcat((collect(r) for r in fgs.chunk_ranges)...) == collect(1:n_leaves)
@test all(!isempty, fgs.chunk_ranges)

chunk_of_leaf = zeros(Int, n_leaves)
for (c, r) in enumerate(fgs.chunk_ranges), i_leaf in r
    chunk_of_leaf[i_leaf] = c
end
leaf_starts = [first(fgs.targets_by_branch[b]) for b in fgs.source_tree.leaf_index]

n_intra = 0
n_cross = 0
for i_leaf in 1:n_leaves
    # partition covers every nonempty segment exactly once, offsets match the
    # unsplit path's running i_influence_start
    expected = Tuple{Int,Int}[]
    i_influence_start = 1
    for index in fgs.index_map[i_leaf]
        i_target, _ = fgs.direct_list[index]
        rows = fgs.targets_by_branch[i_target]
        isempty(rows) || push!(expected, (index, i_influence_start))
        i_influence_start += FastMultipole.get_n_bodies(fgs.source_tree.branches[i_target].bodies_index)
    end
    combined = sort(vcat(fgs.scatter_intra[i_leaf], fgs.scatter_cross[i_leaf]))
    @test combined == expected

    # intra segments write only rows owned by the source's own chunk
    for (index, _) in fgs.scatter_intra[i_leaf]
        i_target, _ = fgs.direct_list[index]
        rows = fgs.targets_by_branch[i_target]
        lo = max(searchsortedlast(leaf_starts, first(rows)), 1)
        hi = max(searchsortedlast(leaf_starts, last(rows)), 1)
        @test chunk_of_leaf[lo] == chunk_of_leaf[i_leaf]
        @test chunk_of_leaf[hi] == chunk_of_leaf[i_leaf]
        n_intra += 1
    end
    n_cross += length(fgs.scatter_cross[i_leaf])
end
@test n_intra > 0                  # premise: both classes non-vacuous
@test n_cross > 0

# purity: identical across repeated construction
fgs_re = make_fgs(:chunked)
@test fgs_re.chunk_ranges == fgs.chunk_ranges
@test fgs_re.scatter_intra == fgs.scatter_intra
@test fgs_re.scatter_cross == fgs.scatter_cross

function seed_state!(s)
    s.self_matrices.rhs .= sin.(eachindex(s.self_matrices.rhs))
    s.nonself_matrices.rhs .= 0
    s.old_influence_storage .= 0
    s.strengths .= 0
    return nothing
end

run_sweep!(s) = FastMultipole.gs_sweep!(s.strengths, s.self_matrices,
    s.leaf_lu_cache, s.self_matrices.rhs, s.nonself_matrices,
    s.old_influence_storage, s.source_tree, s.target_tree,
    s.strengths_by_leaf, s.index_map, s.direct_list, s.targets_by_branch,
    s, false)

# --- 2. single-chunk equivalence: nchunks=1 chunked == lexicographic --------
fgs_one = make_fgs(:chunked; chunks=1)
@test length(fgs_one.chunk_ranges) == 1
@test all(isempty, fgs_one.scatter_cross)     # one chunk: everything intra
fgs_lex = make_fgs(:lexicographic)
seed_state!(fgs_one); seed_state!(fgs_lex)
run_sweep!(fgs_one)
run_sweep!(fgs_lex)
@test fgs_one.strengths == fgs_lex.strengths                    # bitwise
@test fgs_one.self_matrices.rhs == fgs_lex.self_matrices.rhs    # bitwise
@test any(!iszero, fgs_one.strengths)                           # non-vacuous

# --- 3. lexicographic regression guard: replica of the historical loop ------
fgs_lex2 = make_fgs(:lexicographic)
seed_state!(fgs_lex2)
for (i_leaf, i_branch) in enumerate(fgs_lex2.source_tree.leaf_index)
    leaf_strengths = view(fgs_lex2.strengths, fgs_lex2.strengths_by_leaf[i_leaf])
    FastMultipole.solve_leaf!(leaf_strengths, fgs_lex2.self_matrices,
        fgs_lex2.leaf_lu_cache, i_leaf)
    FastMultipole.update_nonself_influence!(fgs_lex2.self_matrices.rhs,
        fgs_lex2.strengths, fgs_lex2.nonself_matrices,
        fgs_lex2.old_influence_storage, i_leaf, fgs_lex2.source_tree,
        fgs_lex2.target_tree, fgs_lex2.strengths_by_leaf, fgs_lex2.index_map,
        fgs_lex2.direct_list, fgs_lex2.targets_by_branch)
end
@test fgs_lex2.strengths == fgs_lex.strengths
@test fgs_lex2.self_matrices.rhs == fgs_lex.self_matrices.rhs

# --- 4. threaded sweep == serial chunk-major reference ----------------------
fgs_ref = make_fgs(:chunked)
seed_state!(fgs); seed_state!(fgs_ref)
run_sweep!(fgs)     # uses Threads.@threads at whatever nthreads this session has
# serial reference: chunk-major GS with immediate intra scatter, then
# deferred cross scatter in ascending leaf order — plain serial loops
for r in fgs_ref.chunk_ranges
    for i_leaf in r
        leaf_strengths = view(fgs_ref.strengths, fgs_ref.strengths_by_leaf[i_leaf])
        FastMultipole.solve_leaf!(leaf_strengths, fgs_ref.self_matrices,
            fgs_ref.leaf_lu_cache, i_leaf)
        FastMultipole.compute_nonself_products!(fgs_ref.strengths,
            fgs_ref.nonself_matrices, fgs_ref.old_influence_storage, i_leaf,
            fgs_ref.strengths_by_leaf)
        FastMultipole.scatter_nonself_influence_partial!(fgs_ref.self_matrices.rhs,
            fgs_ref.nonself_matrices, fgs_ref.old_influence_storage, i_leaf,
            fgs_ref.scatter_intra[i_leaf], fgs_ref.target_tree,
            fgs_ref.direct_list, fgs_ref.targets_by_branch)
    end
end
for r in fgs_ref.chunk_ranges, i_leaf in r
    FastMultipole.scatter_nonself_influence_partial!(fgs_ref.self_matrices.rhs,
        fgs_ref.nonself_matrices, fgs_ref.old_influence_storage, i_leaf,
        fgs_ref.scatter_cross[i_leaf], fgs_ref.target_tree,
        fgs_ref.direct_list, fgs_ref.targets_by_branch)
end
@test fgs.strengths == fgs_ref.strengths                    # bitwise
@test fgs.self_matrices.rhs == fgs_ref.self_matrices.rhs    # bitwise
@test any(!iszero, fgs.strengths)                           # non-vacuous

# --- 5. + 6. fixed-iteration solves: repeatability, cross-process -t 1/4 ----
fgs_solve = make_fgs(:chunked)
function cold_fixed_solve!(fgs_obj)
    for i in eachindex(system.bodies)
        body = system.bodies[i]
        system.bodies[i] = typeof(body)(body.position, body.radius, 0.0)
    end
    residuals = Float64[]
    FastMultipole.solve!(system, fgs_obj; scalar_potential=true, gradient=false,
        max_iterations=6, inner_iterations=2, tolerance=-1.0,
        reverse_pass=false, final_update=false, verbose=false,
        callback=(_, residual) -> push!(residuals, residual))
    strengths = [body.strength for body in system.bodies]
    return collect(reinterpret(UInt64, residuals)),
           collect(reinterpret(UInt64, strengths))
end
ref_res, ref_str = cold_fixed_solve!(fgs_solve)
for _ in 1:3
    res, str = cold_fixed_solve!(fgs_solve)
    @test res == ref_res
    @test str == ref_str
end

# cross-process thread-count invariance: -t 1 and -t 4 subprocesses rebuild
# the identical fixture and solve; results must match the parent bit-for-bit
helper = joinpath(@__DIR__, "fgs_chunked_threadcheck.jl")
for t in (1, 4)
    out = tempname()
    cmd = `$(Base.julia_cmd()) -t $t --startup-file=no --project=$(Base.active_project()) $helper $out`
    run(cmd)
    lines = readlines(out)
    sub_res = parse.(UInt64, split(lines[1]))
    sub_str = parse.(UInt64, split(lines[2]))
    @test sub_res == ref_res
    @test sub_str == ref_str
    rm(out; force=true)
end

# iteration health: short fixed-iteration histories exist and stay finite for
# both orders (convergence at campaign scale is gated by the 021 A/B, not here
# — see the analogous note in fgs_coloring_test.jl)
function short_history(fgs_obj)
    for i in eachindex(system.bodies)
        body = system.bodies[i]
        system.bodies[i] = typeof(body)(body.position, body.radius, 0.0)
    end
    residuals = Float64[]
    FastMultipole.solve!(system, fgs_obj; scalar_potential=true, gradient=false,
        max_iterations=6, inner_iterations=2, tolerance=-1.0,
        reverse_pass=false, final_update=false, verbose=false,
        callback=(_, residual) -> push!(residuals, residual))
    return residuals
end
res_chunked = short_history(make_fgs(:chunked))
res_lex = short_history(make_fgs(:lexicographic))
@test length(res_chunked) == length(res_lex) == 6
@test all(isfinite, res_chunked)
@test all(isfinite, res_lex)

# --- guards ------------------------------------------------------------------
@test_throws ArgumentError make_fgs(:chunked; chunks=0)
big = make_fgs(:chunked; chunks=10_000)       # clamps to n_leaves
@test length(big.chunk_ranges) == n_leaves
@test all(r -> length(r) == 1, big.chunk_ranges)
lex = make_fgs(:lexicographic)
@test isempty(lex.chunk_ranges)
@test isempty(lex.scatter_intra)
@test isempty(lex.scatter_cross)

end
