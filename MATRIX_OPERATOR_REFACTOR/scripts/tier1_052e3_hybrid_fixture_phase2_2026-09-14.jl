# 052e.3 Phase 2 — simulation lift parity: S1/S2/S3 free-running sims at L2
# Prereg (LOCKED 2026-09-12): 052e3-hybrid-fixture-preregistration-2026-09-12.md
#   (Phase 2 design registered there; no new prereg)
# Supersessions (pre-run): 052e3-hybrid-fixture-supersession-2026-09-12.md
# Harness-realization choices recorded pre-launch:
#   052e3-phase2-realization-2026-09-14.md
#
# Sims (same wing, L2, identical kinematics/dt/steps, simulate! free-running):
#   S1 = doublet-panel wake + DirectWakePotential            (reference)
#   S2 = doublet-panel wake + GreenReconstruction(:area_mean)
#   S3 = particle wake (production shedding OverlapPPS(2.4,2)) + GR
# Metrics over settled window = last 50% of steps, all three pairs:
#   Delta_mean and Delta_max of C_L(t) (Kutta-Joukowski CL, addendum R3).
#   No pass/fail gates: raw values reported next to registered a priori
#   estimates (P3: S2-vs-S1 ~<1e-2; P4: S3 pairs 2e-2..6e-2), Ryan rules.
#
# Smoke: T1_SMOKE=1 julia --project=$SILO/FLOWPanel.jl --threads=N <this file>
# Registered run: same command without T1_SMOKE; outputs under
#   MATRIX_OPERATOR_REFACTOR/data/052e3-phase2/.

import FLOWPanel as pnl
using StaticArrays, Printf, Statistics, LinearAlgebra, SHA

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl", "examples", "pitching_wing.jl"))

const SMOKE = get(ENV, "T1_SMOKE", "0") == "1"
const DATADIR = SMOKE ?
    get(ENV, "T1_SMOKE_DIR", joinpath(tempdir(), "tier1_052e3_phase2_smoke")) :
    joinpath(HERE, "..", "data", "052e3-phase2")
mkpath(DATADIR)

# ---------------- fixture constants (locked; Phase-1/addendum-identical) -------
const C_FIX = 0.76
const B_FIX = 2.7
const THICK = 0.12
const AOA = 30.0
const UMAG = 1.0
const VINF = SVector{3}(UMAG, 0.0, 0.0)
Uinf_f(t) = VINF
noman!(frames, systems, wakes, t) = nothing

# L2 of the stage-1 ladder (registered level for Phase 2)
const LEVEL = SMOKE ? (41, 5, 3) : (121, 10, 7)

# Harness-realization choices (recorded pre-launch; mirror 052e.2a addendum
# Phase B production defaults, addendum R2):
const DT = 0.5 * C_FIX / UMAG          # production c_per_dt = 0.5
const NSAMP = SMOKE ? 8 : 61           # solve samples (60 sheds ~ 30c wake)
const W1_NWAKEROWS = NSAMP + 2         # panel wake retains all rows
const W2_NWAKEROWS = SMOKE ? 2 : 4     # short panel buffer -> mostly particles
const W2_MAXPART = SMOKE ? 5000 : 200_000
const DAS = 0.05 * C_FIX               # production das convention (addendum R11)

const DIRECT = pnl.DirectBackend()
const PIVOT = SVector{3}(0.25 * C_FIX, 0.0, 0.0)

# Settled window = last 50% of the NSAMP solve samples
const NWIN = NSAMP - div(NSAMP, 2)     # 31 of 61 (samples i_step 30..60)

const SIMS = (
    (id="S1", arm=:W1, gr=false, form=() -> pnl.DirectWakePotential()),
    (id="S2", arm=:W1, gr=true, form=() -> pnl.GreenReconstruction(gauge=:area_mean)),
    (id="S3", arm=:W2, gr=true, form=() -> pnl.GreenReconstruction(gauge=:area_mean)),
)

# ---------------- body / shedding helpers (Phase-1-identical) ------------------
function make_capped_wing()
    n_airfoil, n_span, n_endcap = LEVEL
    wing = build_pitching_wing_body(C_FIX, B_FIX; n_span, n_airfoil, n_endcap,
        thickness=THICK, semiinfinite_wake=false)
    pnl.calc_normals!(wing)
    pnl.calc_controlpoints!(wing)
    return wing
end

struct ShedInfo
    isurf::Vector{Int}
    icol::Vector{Int}
    p_up::Vector{Int}
    p_lo::Vector{Int}
    y::Vector{Float64}
    dy::Vector{Float64}
    te_nodes::Set{Int}
end

function shed_info(body)
    isurfs = Int[]; icols = Int[]; pups = Int[]; plos = Int[]
    ys = Float64[]; dys = Float64[]
    tenodes = Set{Int}()
    for (isurf, sh) in enumerate(body.shedding)
        for i in axes(sh, 2)
            pi = sh[1, i]
            n1 = body.cells[sh[2, i], pi]
            n2 = body.cells[sh[3, i], pi]
            push!(ys, (body.nodes[2, n1] + body.nodes[2, n2]) / 2)
            push!(dys, abs(body.nodes[2, n2] - body.nodes[2, n1]))
            push!(tenodes, n1, n2)
            pj = sh[4, i]
            if pj != -1
                push!(tenodes, body.cells[sh[5, i], pj], body.cells[sh[6, i], pj])
            end
            push!(isurfs, isurf); push!(icols, i); push!(pups, pi); push!(plos, pj)
        end
    end
    return ShedInfo(isurfs, icols, pups, plos, ys, dys, tenodes)
end

gamma_of(strength, info::ShedInfo) =
    [strength[info.p_up[k], end] -
     (info.p_lo[k] == -1 ? 0.0 : strength[info.p_lo[k], end])
     for k in eachindex(info.p_up)]

gtot_of(G, info) = sum(G .* info.dy)
cl_of(G, info) = 2 * gtot_of(G, info) / (UMAG * B_FIX * C_FIX)

function kutta_xcheck(body, info, strength_cap)
    saved = copy(body.strength)
    body.strength .= strength_cap
    err = 0.0
    try
        for k in eachindex(info.p_up)
            si, sj = pnl._get_wakestrength_mu(body, info.icol[k], info.isurf[k])
            Gk = strength_cap[info.p_up[k], end] -
                 (info.p_lo[k] == -1 ? 0.0 : strength_cap[info.p_lo[k], end])
            err = max(err, abs(Gk - (si - sj)))
        end
    catch
        err = NaN
    end
    body.strength .= saved
    return err
end

# ---------------- telemetry ----------------------------------------------------
np_of(wake) = hasproperty(wake, :pfield) ?
    (try Int(pnl.FLOWVPM.get_np(wake.pfield)) catch; -1 end) : -1

mutable struct RunStore
    steps::Vector{Any}
    cls::Vector{Float64}
    gr::Vector{Any}
    cap::Any
end
RunStore() = RunStore([], Float64[], [], nothing)

function make_callback(body, wake, info, store::RunStore; gr::Bool, capture_at::Int)
    return (nt) -> begin
        st = nt.formulation_state
        G = gamma_of(body.strength, info)
        all(isfinite, body.strength) || error("non-finite strengths at step $(nt.i_step)")
        push!(store.cls, cl_of(G, info))
        push!(store.steps, (i=nt.i_step, gtot=gtot_of(G, info),
            cl=store.cls[end], maxs=maximum(abs, body.strength),
            np=np_of(wake)))
        if gr
            push!(store.gr, (i=nt.i_step, q=copy(st.green.q), sigma=copy(st.sigma),
                lambda=pnl._green_lambda(st.green)))
        end
        if nt.i_step == capture_at
            store.cap = (strength=copy(body.strength),)
        end
        nothing
    end
end

# ---------------- GR-system diagnostics (recorded, not gated; Phase 1 pattern) --
function green_diag(body, grstep)
    a = pnl._panel_areas(body)
    N = length(a)
    Ssig = zeros(N)
    pnl._source_potential!(Ssig, body, grstep.sigma, DIRECT)
    Bq = zeros(N)
    pnl._with_green_scratch(body) do
        pnl._green_B_product!(Bq, body, grstep.q, DIRECT)
    end
    res = norm(grstep.q .- Bq .+ a .* grstep.lambda .- Ssig) / max(norm(Ssig), eps())
    gd = abs(dot(a, grstep.q)) / max(norm(a) * norm(grstep.q), eps())
    return (; res, gd, lambda=grstep.lambda)
end

# ---------------- one simulation ------------------------------------------------
function run_sim(sim)
    runid = "$(sim.id)-L2"
    body = make_capped_wing()
    info = shed_info(body)
    frames = pitching_wing_frame(body, PIVOT, deg2rad(AOA))
    set_wake_Das!(body, VINF; magnitude=DAS)   # after frames: production convention
    local wake
    if sim.arm == :W1
        wake = pnl.PanelWake(body; nwakerows=W1_NWAKEROWS,
            include_final_filament=false)
        pnl.update_TE!(wake, body)
    else
        wake = pnl.PanelParticleWake(body; nwakerows=W2_NWAKEROWS,
            max_particles=W2_MAXPART,
            method_trailing=pnl.OverlapPPS(2.4, 2),
            method_unsteady=pnl.OverlapPPS(2.4, 2))
    end
    store = RunStore()
    cb = make_callback(body, wake, info, store; gr=sim.gr, capture_at=NSAMP - 1)
    trange = range(0.0, step=DT, length=NSAMP)
    t_el = @elapsed pnl.simulate!((body,), (wake,), frames, noman!, Uinf_f, trange;
        body_solvers=(pnl.Backslash(body),), backend=DIRECT, path=nothing,
        formulation=sim.form(), step_telemetry_callback=cb, verbose=false)
    cap = store.cap
    cap === nothing && error("$runid: capture step never fired")
    length(store.cls) == NSAMP || error("$runid: expected $NSAMP samples, got $(length(store.cls))")
    kx = kutta_xcheck(body, info, cap.strength)
    # Phase-B steadiness (recorded, not gated): |Gtot drift| over last 5 steps
    gts = [s.gtot for s in store.steps]
    drift = length(gts) >= 6 ?
        abs(gts[end] - gts[end-5]) / max(abs(gts[end]), eps()) : NaN
    gd = sim.gr ? green_diag(body, store.gr[end]) :
        (; res=NaN, gd=NaN, lambda=NaN)
    open(joinpath(DATADIR, "steps_$(runid).csv"), "w") do io
        println(io, "i,t,CL,Gamma_tot,maxabs_strength,np")
        for (k, s) in enumerate(store.steps)
            @printf(io, "%d,%.9e,%.9e,%.9e,%.9e,%d\n",
                s.i, trange[k], s.cl, s.gtot, s.maxs, s.np)
        end
    end
    if sim.gr
        open(joinpath(DATADIR, "grdiag_$(runid).csv"), "w") do io
            println(io, "i,lambda")
            for r in store.gr
                @printf(io, "%d,%.9e\n", r.i, r.lambda)
            end
        end
    end
    @printf("  %-8s N=%-6d CLend=%+.5f drift=%.1e np=%d kx=%.1e grres=%.1e grgd=%.1e t=%.0fs\n",
        runid, body.ncells, store.cls[end], drift, store.steps[end].np, kx,
        gd.res, gd.gd, t_el)
    flush(stdout)
    return (; runid, N=body.ncells, cls=store.cls, steps=store.steps,
        drift, kutta_x=kx, grdiag=gd, t_el)
end

# ---------------- pair metrics (registered definitions, prereg lines 96-97) ------
# Delta_mean = |mean(CL_A) - mean(CL_B)| / |mean(CL_B)|
# Delta_max  = max_t |CL_A(t) - CL_B(t)| / |mean(CL_B)|
function cl_pair(recA, recB)
    wA = recA.cls[end-NWIN+1:end]
    wB = recB.cls[end-NWIN+1:end]
    ref = abs(mean(wB))
    d_mean = abs(mean(wA) - mean(wB)) / max(ref, eps())
    d_max = maximum(abs.(wA .- wB)) / max(ref, eps())
    return (; d_mean, d_max, raw_mean=abs(mean(wA) - mean(wB)),
        raw_max=maximum(abs.(wA .- wB)), clA=mean(wA), clB=mean(wB))
end

# ---------------- main ------------------------------------------------------------
lines = String[]
say(s) = (push!(lines, s); println(s); flush(stdout))

println("== 052e.3 Phase 2 $(SMOKE ? "(SMOKE, unregistered)" : "(REGISTERED)") ==")
println("level=$(LEVEL) NSAMP=$NSAMP DT=$DT W1rows=$W1_NWAKEROWS W2rows=$W2_NWAKEROWS W2maxpart=$W2_MAXPART NWIN=$NWIN")
println("datadir=$DATADIR threads=$(Threads.nthreads())")
flush(stdout)

recs = Dict{String,Any}()
try
    for sim in SIMS
        recs[sim.id] = run_sim(sim)
        GC.gc()
    end

    open(joinpath(DATADIR, "cl_timeseries_L2.csv"), "w") do io
        println(io, "i,t,CL_S1,CL_S2,CL_S3")
        trange = range(0.0, step=DT, length=NSAMP)
        for k in 1:NSAMP
            @printf(io, "%d,%.9e,%.9e,%.9e,%.9e\n", k - 1, trange[k],
                recs["S1"].cls[k], recs["S2"].cls[k], recs["S3"].cls[k])
        end
    end

    say(@sprintf("N=%d  settled window = last %d of %d samples", recs["S1"].N, NWIN, NSAMP))
    for r in ("S1", "S2", "S3")
        w = recs[r].cls[end-NWIN+1:end]
        say(@sprintf("%s: CL mean=%+.6f min=%+.6f max=%+.6f (window) CLend=%+.6f drift=%.2e np_end=%d kx=%.1e",
            r, mean(w), minimum(w), maximum(w), recs[r].cls[end],
            recs[r].drift, recs[r].steps[end].np, recs[r].kutta_x))
    end
    for (aid, bid, est) in (("S2", "S1", "P3: ~<1e-2"),
                            ("S3", "S2", "P4: 2e-2..6e-2"),
                            ("S3", "S1", "P4: 2e-2..6e-2"))
        p = cl_pair(recs[aid], recs[bid])
        say(@sprintf("%s vs %s: Delta_mean=%.4e Delta_max=%.4e  (unnormalized: %.4e, %.4e)  [a priori %s]",
            aid, bid, p.d_mean, p.d_max, p.raw_mean, p.raw_max, est))
    end
    for r in ("S2", "S3")
        gd = recs[r].grdiag
        say(@sprintf("%s GR diag (final step, recorded): res=%.2e gd=%.2e lam=%+.3e",
            r, gd.res, gd.gd, gd.lambda))
    end
finally
    function gitinfo(path)
        try
            sha = strip(read(`git -C $path rev-parse HEAD`, String))
            dirty = !isempty(strip(read(`git -C $path status --porcelain`, String)))
            dh = dirty ? bytes2hex(sha256(read(`git -C $path diff`)))[1:12] : ""
            "$sha $(dirty ? "DIRTY tracked-diff=$dh" : "clean")"
        catch
            "no-git (silo copy; see silo_provenance block below)"
        end
    end
    provfile = joinpath(HERE, "..", "..", "..", "silo_provenance.txt")
    script_sha = bytes2hex(sha256(read(@__FILE__)))[1:12]
    open(joinpath(DATADIR, "gates.txt"), "w") do io
        println(io, "052e.3 Phase 2 $(SMOKE ? "SMOKE (UNREGISTERED — values meaningless)" : "REGISTERED")")
        println(io, "date=$(chomp(read(`date +%Y-%m-%d`, String))) julia=$(VERSION) threads=$(Threads.nthreads()) script_sha256=$script_sha")
        println(io, "FLOWPanel: $(gitinfo(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl")))")
        println(io, "FastMultipole: $(gitinfo(joinpath(HERE, "..", "..")))")
        if isfile(provfile)
            println(io, "silo_provenance:")
            foreach(l -> println(io, "  " * l), eachline(provfile))
        end
        println(io, "fixture: AOA=$(AOA)deg |U|=$UMAG level=$(LEVEL) NSAMP=$NSAMP DT=$DT W1rows=$W1_NWAKEROWS W2rows=$W2_NWAKEROWS W2maxpart=$W2_MAXPART shed=OverlapPPS(2.4,2) NWIN=$NWIN")
        println(io, "sims: S1=W1+DirectWakePotential S2=W1+GreenReconstruction(:area_mean) S3=W2(particles)+GR | simulate! free-running, identical kinematics/dt/steps")
        println(io, "metrics: settled window = last 50% of samples; registered Delta_mean=|mean(CL_A)-mean(CL_B)|/|mean(CL_B)|, Delta_max=max_t|CL_A-CL_B|/|mean(CL_B)|; Kutta-Joukowski CL; pairs S2vS1, S3vS2, S3vS1")
        println(io, "reporting: no pass/fail gates (prereg) — raw values next to a priori estimates P3 (S2-vs-S1 ~<1e-2), P4 (S3 pairs 2e-2..6e-2); Ryan rules")
        for l in lines
            println(io, l)
        end
    end
    println("\n===== 052e.3 PHASE 2 $(SMOKE ? "SMOKE" : "MEASUREMENTS") =====")
    foreach(println, lines)
    println("outputs: $DATADIR")
    println("=========================================")
end
