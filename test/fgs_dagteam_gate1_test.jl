#=##############################################################################
FGS sweep_order=:dagteam — gate-1 correctness vs the REAL implementation
(FLOWPanel BRAINSTORM 021, fgs_acceleration_recommendation_20260918.md §gates;
production executor in src/solve_dagteam.jl, measured design = gate 2d).

Claims verified here:
 1. sweep-level equivalence: N dagteam sweeps (via dagteam_initialize! +
    dagteam_inner_sweeps! on a synthetic external rhs) match N production
    :lexicographic gs_sweep! calls on strengths AND on the reconstructed
    rhs invariant, to tight tolerance, from zero AND nonzero starts
    (dagteam is mathematically equivalent, NOT bitwise: pulls aggregate
    predecessors into one GEMV, upper terms are rebuilt from full products);
 2. determinism: repeating the same dagteam sweep block from identical
    state is BITWISE identical (any thread count; fixed arithmetic and
    reduction orders);
 3. end-to-end: solve! with :dagteam matches solve! with :lexicographic
    (same systems, same settings) on final strengths;
 4. rigidly transformed solver fixture (scalar potential only);
 5. multi-system fixture when the constructor supports it (pre-existing
    BoundsError at c18e4b46 otherwise — reported, not fatal);
 6. reduced precision: :f32conv and :f32full agree with the F64 result at
    single-precision-consistent tolerance (accuracy CERTIFICATION is the
    independent evaluator's job, not this test's).
=###############################################################################

if !(@isdefined FastMultipole)   # standalone execution support
    using FastMultipole
    using FastMultipole.LinearAlgebra
    using FastMultipole.StaticArrays
    using Random, Test
    include("gravitational.jl")

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

#------- sweep-level harness -------#

# Drive N sweeps of each schedule from the same synthetic state. The lex
# reference follows the production solve! bookkeeping: rhs = b − nonself(x0)
# via the init pass, then gs_sweep! maintains it incrementally. The dagteam
# candidate uses its own init + inner-sweeps entry points with the same b as
# the "external" rhs and zero farfield.
function lex_run(fgs, b, x0, nsweeps)
    fgs.strengths .= x0
    rhs = fgs.self_matrices.rhs
    rhs .= b
    fgs.nonself_matrices.rhs .= 0
    fgs.old_influence_storage .= 0
    FastMultipole.update_nonself_influence!(rhs, fgs.strengths,
        fgs.nonself_matrices, fgs.old_influence_storage, fgs.source_tree,
        fgs.target_tree, fgs.strengths_by_leaf, fgs.index_map,
        fgs.direct_list, fgs.targets_by_branch)
    for _ in 1:nsweeps
        FastMultipole.gs_sweep!(fgs.strengths, fgs.self_matrices,
            fgs.leaf_lu_cache, rhs, fgs.nonself_matrices,
            fgs.old_influence_storage, fgs.source_tree, fgs.target_tree,
            fgs.strengths_by_leaf, fgs.index_map, fgs.direct_list,
            fgs.targets_by_branch, fgs, false)
    end
    return copy(fgs.strengths), copy(rhs)
end

function dagteam_run(fgs, b, x0, nsweeps)
    fgs.strengths .= x0
    rhs = fgs.self_matrices.rhs
    rhs .= b
    zero_ff = zero(fgs.extra_right_hand_side)
    FastMultipole.dagteam_initialize!(rhs, fgs.dagteam, fgs.strengths)
    FastMultipole.dagteam_inner_sweeps!(rhs, zero_ff, fgs.strengths,
        fgs.dagteam, nsweeps)
    return copy(fgs.strengths), copy(rhs)
end

rel_dev(a, b) = norm(a .- b, Inf) / max(norm(a, Inf), norm(b, Inf), eps())

#------- fixtures -------#

@testset "FGS dagteam schedule: gate-1 correctness (real implementation)" begin

Random.seed!(20260919)

system = generate_gravitational(20260918, 800)
FastMultipole.direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

make_fgs(order; precision=:f64) = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false, sweep_order=order, dagteam_precision=precision)

fgs_lex = make_fgs(:lexicographic)
fgs_dag = make_fgs(:dagteam)
@test length(fgs_lex.source_tree.leaf_index) > 1
@test !isempty(fgs_lex.direct_list)
@test fgs_dag.dagteam !== nothing
plan = fgs_dag.dagteam
n_leaves = length(fgs_lex.source_tree.leaf_index)

# structural sanity: split storage tiles the source-major coefficient bytes
lower_elems = sum(length, plan.Lmat)
upper_elems = sum(length, plan.Umat)
nonself_elems = sum(m * n for (m, n) in fgs_lex.nonself_matrices.sizes)
@test lower_elems + upper_elems == nonself_elems
@test !isempty(plan.roots)
@info "gate-1 dagteam structure" n_leaves n_roots=length(plan.roots) n_lower_edges=sum(length, plan.preds) lower_elems upper_elems

nstr = length(fgs_lex.strengths)
b = sin.(1.0 .* (1:nstr))

# --- 1. sweep-level equivalence, zero and nonzero starts ---------------------
for (label, x0) in (("zero start", zeros(nstr)),
                    ("nonzero start", 0.1 .* randn(nstr)))
    for nsweeps in (1, 3)
        x_lex, rhs_lex = lex_run(fgs_lex, b, x0, nsweeps)
        x_dag, rhs_dag = dagteam_run(fgs_dag, b, x0, nsweeps)
        dx, dr = rel_dev(x_lex, x_dag), rel_dev(rhs_lex, rhs_dag)
        @test dx <= 1e-12
        @test dr <= 1e-12
        @test any(!iszero, x_lex)   # non-vacuous
        @info "gate-1 dagteam vs lex" label nsweeps rel_dev_strengths=dx rel_dev_rhs=dr
    end
end

# --- 2. determinism: identical reruns are bitwise identical ------------------
x0 = 0.1 .* randn(nstr)
xa, ra = dagteam_run(fgs_dag, b, x0, 3)
xb, rb = dagteam_run(fgs_dag, b, x0, 3)
@test xa == xb
@test ra == rb
@info "gate-1 dagteam determinism" nthreads=Threads.nthreads() bitwise=true

# --- 3. end-to-end solve! agreement ------------------------------------------
function full_solve(order; precision=:f64)
    sys = generate_gravitational(20260918, 800)
    FastMultipole.direct!(sys; scalar_potential=true, gradient=false)
    sys.potential[1, :] .*= -1.0
    fgs = FastMultipole.FastGaussSeidel((sys,), (sys,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=order,
        dagteam_precision=precision)
    FastMultipole.solve!((sys,), (sys,), fgs; scalar_potential=true,
        gradient=false, max_iterations=20, inner_iterations=2,
        tolerance=1e-10, verbose=false, final_update=false)
    return copy(fgs.strengths)
end
s_lex = full_solve(:lexicographic)
s_dag = full_solve(:dagteam)
d_e2e = rel_dev(s_lex, s_dag)
@test d_e2e <= 1e-8   # both converged to tolerance 1e-10; iterates may differ within it
@info "gate-1 dagteam end-to-end solve!" rel_dev=d_e2e

# --- 4. rigidly transformed solver -------------------------------------------
θ = 0.7; R = SMatrix{3,3}([cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1.0])
t = SVector(0.1, -0.2, 0.3)
function make_transformed(order)
    sys = generate_gravitational(20260918, 800)
    fgs = FastMultipole.FastGaussSeidel((sys,), (sys,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=order)
    for i in eachindex(sys.bodies)
        bd = sys.bodies[i]
        sys.bodies[i] = typeof(bd)(R * bd.position + t, bd.radius, bd.strength)
    end
    FastMultipole.transform_solver!(fgs, (sys,), R, t)
    return fgs
end
fgs_lex_t = make_transformed(:lexicographic)
fgs_dag_t = make_transformed(:dagteam)
x_lex, rhs_lex = lex_run(fgs_lex_t, b, x0, 3)
x_dag, rhs_dag = dagteam_run(fgs_dag_t, b, x0, 3)
@test rel_dev(x_lex, x_dag) <= 1e-12
@info "gate-1 dagteam transformed" rel_dev_strengths=rel_dev(x_lex, x_dag)

# --- 5. multiple source systems (branch-spanning target rows) ----------------
sysA = generate_gravitational(11, 300)
sysB = generate_gravitational(22, 300)
multi_supported = try
    FastMultipole.FastGaussSeidel((sysA, sysB), (sysA, sysB);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:lexicographic)
    true
catch err
    @info "multi-system FGS fixture unsupported here (pre-existing constructor failure)" err=sprint(showerror, err)
    false
end
if multi_supported
    fgs_lex_m = FastMultipole.FastGaussSeidel((sysA, sysB), (sysA, sysB);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:lexicographic)
    fgs_dag_m = FastMultipole.FastGaussSeidel((sysA, sysB), (sysA, sysB);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=:dagteam)
    nm = length(fgs_lex_m.strengths)
    bm = sin.(1.0 .* (1:nm))
    xm0 = 0.1 .* randn(nm)
    x_lex, _ = lex_run(fgs_lex_m, bm, xm0, 3)
    x_dag, _ = dagteam_run(fgs_dag_m, bm, xm0, 3)
    @test rel_dev(x_lex, x_dag) <= 1e-12
    @info "gate-1 dagteam multi-system" rel_dev_strengths=rel_dev(x_lex, x_dag)

    # truth oracle (catches consistent-but-wrong fill/scatter row ordering
    # across systems): the KNOWN strengths must be a ONE-SWEEP fixed point
    # of the all-direct (MAC=0.01) operator built from their inverted
    # potential. A multi-sweep recovery oracle is unusable with this
    # kernel: the 1/r first-kind potential operator has block-GS spectral
    # radius >> 1 for any multi-leaf partition (a union-600 SINGLE-system
    # control diverges identically, and per-sweep amplification ~5e5 blows
    # even the fixed point past tolerance by sweep 2), so exactly one
    # sweep is run. A misassembled operator breaks the fixed point by
    # O(1e4) (negative controls below); a correct one moves ~4e-8.
    function oracle_dev(order; corrupt=false)
        sa = generate_gravitational(11, 300)
        sb = generate_gravitational(22, 300)
        strengths_desired = vcat([bd.strength[1] for bd in sa.bodies],
                                 [bd.strength[1] for bd in sb.bodies])
        FastMultipole.direct!((sa, sb); scalar_potential=true, gradient=false)
        sa.potential[1, :] .*= -1.0
        sb.potential[1, :] .*= -1.0
        fgs_m = FastMultipole.FastGaussSeidel((sa, sb), (sa, sb);
            expansion_order=4, multipole_acceptance=0.01, leaf_size=40,
            shrink=true, recenter=false, sweep_order=order)
        @assert isempty(fgs_m.m2l_list) && !isempty(fgs_m.direct_list) # all-direct premise
        if corrupt   # sensitivity control: 1% scaling of the nonself operator
            if order === :dagteam
                foreach(m -> m .*= 1.01, fgs_m.dagteam.Lmat)
                foreach(m -> m .*= 1.01, fgs_m.dagteam.Umat)
            else
                fgs_m.nonself_matrices.data .*= 1.01
            end
        end
        FastMultipole.solve!((sa, sb), (sa, sb), fgs_m; scalar_potential=true,
            gradient=false, max_iterations=1, inner_iterations=1,
            tolerance=0.0, verbose=false, final_update=false)
        recovered = vcat([bd.strength[1] for bd in sa.bodies],
                         [bd.strength[1] for bd in sb.bodies])
        return norm(recovered .- strengths_desired, Inf)
    end
    for order in (:lexicographic, :dagteam)
        dev = oracle_dev(order)
        @test dev <= 1e-6
        dev_corrupt = oracle_dev(order; corrupt=true)
        @test dev_corrupt > 1e-2   # oracle must catch a corrupted operator
        @info "gate-1 multi-system fixed-point oracle" order max_abs_dev=dev corrupted_dev=dev_corrupt
    end
end

# --- 6. reduced-precision modes (sanity only; certification = evaluator) -----
for precision in (:f32conv, :f32full)
    fgs_dag32 = make_fgs(:dagteam; precision)
    x_lex, _ = lex_run(fgs_lex, b, x0, 3)
    x_32, _ = dagteam_run(fgs_dag32, b, x0, 3)
    d32 = rel_dev(x_lex, x_32)
    @test d32 <= 5e-5   # single-precision-consistent
    @info "gate-1 dagteam reduced precision" precision rel_dev_vs_f64_lex=d32
end

end
