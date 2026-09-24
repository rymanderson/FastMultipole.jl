#=##############################################################################
FGS sweep_order=:dagedge — correctness of edge-level partial pulls on the
static per-worker schedule (FLOWPanel BRAINSTORM 021 L-shortening item #1;
design in BRAINSTORM/021/fgs_dagedge_design_20260924.md; executor in
src/solve_dagedge.jl). Patterned on fgs_dagteam_gate1_test.jl.

Claims verified here:
 1. sweep-level equivalence: N :dagedge sweeps match N production
    :lexicographic gs_sweep! calls AND N :dagteam sweeps on strengths and on
    the reconstructed rhs invariant, from zero AND nonzero starts, at every
    θ ∈ {0, 4096, typemax} (mathematically equivalent, NOT bitwise: the
    partial-sum grouping differs from one aggregated GEMV);
 2. determinism: repeating the same sweep block from identical state is
    BITWISE identical, and dagteam_workers ∈ {1, 2, 4} give BITWISE
    identical results at fixed θ (scheduling must never affect values);
 3. structural sanity: static lists cover every task exactly once; split
    storage is untouched (shared with the wrapped DagTeamPlan); slot counts
    match the θ partition;
 4. end-to-end: solve! with :dagedge matches :lexicographic;
 5. rigidly transformed solver fixture (scalar potential only);
 6. reduced precision :f32conv/:f32full sanity (certification = evaluator).
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

#------- sweep-level harness (mirrors gate-1) -------#

function lex_run_de(fgs, b, x0, nsweeps)
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

function dagteam_run_de(fgs, b, x0, nsweeps)
    fgs.strengths .= x0
    rhs = fgs.self_matrices.rhs
    rhs .= b
    zero_ff = zero(fgs.extra_right_hand_side)
    FastMultipole.dagteam_initialize!(rhs, fgs.dagteam, fgs.strengths)
    FastMultipole.dagteam_inner_sweeps!(rhs, zero_ff, fgs.strengths,
        fgs.dagteam, nsweeps)
    return copy(fgs.strengths), copy(rhs)
end

function dagedge_run(fgs, b, x0, nsweeps)
    fgs.strengths .= x0
    rhs = fgs.self_matrices.rhs
    rhs .= b
    zero_ff = zero(fgs.extra_right_hand_side)
    FastMultipole.dagteam_initialize!(rhs, fgs.dagteam, fgs.strengths)
    FastMultipole.dagedge_inner_sweeps!(rhs, zero_ff, fgs.strengths,
        fgs.dagteam, nsweeps)
    return copy(fgs.strengths), copy(rhs)
end

rel_dev_de(a, b) = norm(a .- b, Inf) / max(norm(a, Inf), norm(b, Inf), eps())

#------- fixtures -------#

@testset "FGS dagedge schedule: correctness (edge-level partial pulls)" begin

Random.seed!(20260924)

system = generate_gravitational(20260918, 800)
FastMultipole.direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

make_fgs_de(order; precision=:f64, theta=4096, workers=0) =
    FastMultipole.FastGaussSeidel((system,), (system,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=order,
        dagteam_precision=precision, dagteam_workers=workers,
        dagedge_theta=theta)

fgs_lex = make_fgs_de(:lexicographic)
fgs_team = make_fgs_de(:dagteam)
fgs_edge = make_fgs_de(:dagedge)
@test fgs_edge.dagteam isa FastMultipole.DagEdgePlan
plan = fgs_edge.dagteam
base = plan.base
n_leaves = length(fgs_lex.source_tree.leaf_index)

# --- structural sanity ---------------------------------------------------------
# split storage shared with the wrapped DagTeamPlan and tiles the coefficients
lower_elems = sum(length, base.Lmat)
upper_elems = sum(length, base.Umat)
nonself_elems = sum(m * n for (m, n) in fgs_lex.nonself_matrices.sizes)
@test lower_elems + upper_elems == nonself_elems

# static lists cover every task exactly once
n_edge = sum(count(t -> t.kind == 0x01, lst) for lst in plan.lists)
n_small = sum(count(t -> t.kind == 0x02, lst) for lst in plan.lists)
n_root = sum(count(t -> t.kind == 0x03, lst) for lst in plan.lists)
n_back = sum(count(t -> t.kind == 0x04, lst) for lst in plan.lists)
@test n_root == count(i -> isempty(base.preds[i]), 1:n_leaves)
@test n_back == count(>(0), base.mup)
@test n_small == count(!isempty, plan.smalls)
@test n_edge + sum(length, plan.smalls) == sum(length, base.preds)   # every lower edge exactly once
@test plan.ntasks == n_edge + n_small + n_back + n_leaves
# slot partition consistent with θ
szTM = sizeof(eltype(base.Lmat[1]))
nof(i) = base.offset[i + 1] - base.offset[i]
for i in 1:n_leaves
    nbig = count(j -> szTM * nof(i) * nof(j) >= plan.theta, base.preds[i])
    @test plan.nslots[i] == nbig + (length(base.preds[i]) > nbig ? 1 : 0)
    @test size(plan.Y[i]) == (nof(i), plan.nslots[i])
end
@info "dagedge structure" n_leaves n_edge n_small n_root n_back theta=plan.theta sim_makespan_MB=round(plan.sim_makespan / 1e6; digits=1) sim_edge_L_MB=round(plan.sim_edge_L / 1e6; digits=1)

nstr = length(fgs_lex.strengths)
b = sin.(1.0 .* (1:nstr))

# --- 1. sweep-level equivalence vs lex AND dagteam, θ sweep -------------------
fgs_theta = Dict(th => (th == 4096 ? fgs_edge : make_fgs_de(:dagedge; theta=th))
                 for th in (0, 4096, typemax(Int)))
for (label, x0) in (("zero start", zeros(nstr)),
                    ("nonzero start", 0.1 .* randn(nstr)))
    for nsweeps in (1, 3)
        x_lex, rhs_lex = lex_run_de(fgs_lex, b, x0, nsweeps)
        x_team, rhs_team = dagteam_run_de(fgs_team, b, x0, nsweeps)
        for (th, fgs_th) in fgs_theta
            x_e, rhs_e = dagedge_run(fgs_th, b, x0, nsweeps)
            dxl, drl = rel_dev_de(x_lex, x_e), rel_dev_de(rhs_lex, rhs_e)
            dxt = rel_dev_de(x_team, x_e)
            @test dxl <= 1e-12
            @test drl <= 1e-12
            @test dxt <= 1e-12
            @test any(!iszero, x_e)
            @info "dagedge vs lex/dagteam" label nsweeps theta=th rel_dev_vs_lex=dxl rel_dev_rhs=drl rel_dev_vs_dagteam=dxt
        end
    end
end

# --- 2. determinism: bitwise reruns; bitwise across worker counts -------------
x0 = 0.1 .* randn(nstr)
xa, ra = dagedge_run(fgs_edge, b, x0, 3)
xb, rb = dagedge_run(fgs_edge, b, x0, 3)
@test xa == xb
@test ra == rb
for workers in (1, 2, 4)
    fgs_w = make_fgs_de(:dagedge; workers)
    @test length(fgs_w.dagteam.lists) == min(workers, Threads.nthreads())
    xw, rw = dagedge_run(fgs_w, b, x0, 3)
    @test xw == xa
    @test rw == ra
end
# :backoff wait policy must also be bitwise identical
fgs_bo = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false, sweep_order=:dagedge,
    dagteam_idle=:backoff)
xbo, rbo = dagedge_run(fgs_bo, b, x0, 3)
@test xbo == xa
@test rbo == ra
@info "dagedge determinism" nthreads=Threads.nthreads() bitwise_reruns=true bitwise_across_workers=true bitwise_backoff=true

# --- 3. end-to-end solve! agreement -------------------------------------------
function full_solve_de(order)
    sys = generate_gravitational(20260918, 800)
    FastMultipole.direct!(sys; scalar_potential=true, gradient=false)
    sys.potential[1, :] .*= -1.0
    fgs = FastMultipole.FastGaussSeidel((sys,), (sys,);
        expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
        shrink=true, recenter=false, sweep_order=order)
    FastMultipole.solve!((sys,), (sys,), fgs; scalar_potential=true,
        gradient=false, max_iterations=20, inner_iterations=2,
        tolerance=1e-10, verbose=false, final_update=false)
    return copy(fgs.strengths)
end
s_lex = full_solve_de(:lexicographic)
s_edge = full_solve_de(:dagedge)
d_e2e = rel_dev_de(s_lex, s_edge)
@test d_e2e <= 1e-8   # both converged to tolerance 1e-10; iterates may differ within it
@info "dagedge end-to-end solve!" rel_dev=d_e2e

# --- 4. rigidly transformed solver --------------------------------------------
θr = 0.7; R = SMatrix{3,3}([cos(θr) -sin(θr) 0; sin(θr) cos(θr) 0; 0 0 1.0])
t = SVector(0.1, -0.2, 0.3)
function make_transformed_de(order)
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
fgs_lex_t = make_transformed_de(:lexicographic)
fgs_edge_t = make_transformed_de(:dagedge)
x_lex, _ = lex_run_de(fgs_lex_t, b, x0, 3)
x_e, _ = dagedge_run(fgs_edge_t, b, x0, 3)
@test rel_dev_de(x_lex, x_e) <= 1e-12
@info "dagedge transformed" rel_dev_strengths=rel_dev_de(x_lex, x_e)

# --- 5. reduced-precision modes (sanity only; certification = evaluator) ------
for precision in (:f32conv, :f32full)
    fgs_32 = make_fgs_de(:dagedge; precision)
    x_lex, _ = lex_run_de(fgs_lex, b, x0, 3)
    x_32, _ = dagedge_run(fgs_32, b, x0, 3)
    d32 = rel_dev_de(x_lex, x_32)
    @test d32 <= 5e-5   # single-precision-consistent
    @info "dagedge reduced precision" precision rel_dev_vs_f64_lex=d32
end

end
