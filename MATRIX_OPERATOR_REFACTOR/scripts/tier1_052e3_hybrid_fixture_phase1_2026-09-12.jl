# 052e.3 Phase 1 — prescribed panel wake: solve parity, Route D vs Route GR
# Prereg (LOCKED 2026-09-12): 052e3-hybrid-fixture-preregistration-2026-09-12.md
# Supersession (pre-run, metric addition): 052e3-hybrid-fixture-supersession-2026-09-12.md
#
# Routes:
#   D  = pnl.DirectWakePotential()                  (direct q_f to RHS, no mean removed)
#   GR = pnl.GreenReconstruction(gauge=:area_mean)  (Householder-reduced trace to RHS)
# Comparisons per level (supersession H1):
#   SC : self-consistent — each route runs the Phase-A fixed point to TOL_A
#   FZ : frozen-wake     — GR solved once on D's converged frozen wake
# Metrics: M1 raw E_mu, M1a aligned E_mu, M1b constant offset vs <q_f>_A,
#          M2 Gamma(y) dy-weighted rms/max (+ dGtot, dCL). No pass/fail gates.
#
# Smoke: T1_SMOKE=1 julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl \
#            --threads=4 MATRIX_OPERATOR_REFACTOR/scripts/tier1_052e3_hybrid_fixture_phase1_2026-09-12.jl
# Registered run: same command without T1_SMOKE, nohup-detached; outputs under
#   MATRIX_OPERATOR_REFACTOR/data/052e3-phase1/.

import FLOWPanel as pnl
using StaticArrays, Printf, Statistics, LinearAlgebra, SHA

const HERE = @__DIR__
include(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl", "examples", "pitching_wing.jl"))

const SMOKE = get(ENV, "T1_SMOKE", "0") == "1"
const DATADIR = SMOKE ?
    get(ENV, "T1_SMOKE_DIR", joinpath(tempdir(), "tier1_052e3_phase1_smoke")) :
    joinpath(HERE, "..", "data", "052e3-phase1")
mkpath(DATADIR)

# ---------------- fixture constants (locked; addendum-identical) --------------
const C_FIX = 0.76
const B_FIX = 2.7
const THICK = 0.12
const AOA = 30.0
const UMAG = 1.0
const VINF = SVector{3}(UMAG, 0.0, 0.0)
Uinf_f(t) = VINF

const LEVELS = SMOKE ? [(41, 5, 3), (61, 6, 4)] :
    [(81, 7, 5), (121, 10, 7), (181, 15, 11), (271, 22, 15)]   # L1-L4
const NFREE = SMOKE ? 6 : 40           # wake rows (40 x 0.5c = 20c)
const ROWLEN = 0.5 * C_FIX
const DT = 0.5 * C_FIX / UMAG
const NITER_A = SMOKE ? 50 : 80
const TOL_A = 1e-8
const DAS = 0.05 * C_FIX

const DIRECT = pnl.DirectBackend()
const PIVOT = SVector{3}(0.25 * C_FIX, 0.0, 0.0)

const ROUTES = (
    (id="D", gr=false, form=() -> pnl.DirectWakePotential()),
    (id="GR", gr=true, form=() -> pnl.GreenReconstruction(gauge=:area_mean)),
)

# ---------------- body / shedding helpers (addendum-identical) ----------------
function make_capped_wing(lev)
    n_airfoil, n_span, n_endcap = LEVELS[lev]
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

te_mask(body, info) = [any(body.cells[k, j] in info.te_nodes
                           for k in axes(body.cells, 1))
                       for j in axes(body.cells, 2)]

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

# ---------------- wake prescription / freeze ----------------------------------
function prescribe_wake!(wake, Gam, info)
    step = VINF ./ norm(VINF) .* ROWLEN
    for nodes in wake.nodes
        fr = copy(view(nodes, :, 1, :))
        for r in axes(nodes, 2)
            view(nodes, :, r, :) .= fr .+ (r - 1) .* step
        end
    end
    for k in eachindex(info.icol)
        s = wake.strength[info.isurf[k]]
        for r in 1:NFREE
            s[1, r, info.icol[k]] = Gam[k]
        end
    end
    wake.nwakes[] = NFREE
    return wake
end

function fixedpoint_maneuver(body, wake, info)
    return (frames, systems, wakes, t) -> begin
        Gam = gamma_of(body.strength, info)
        pnl.update_TE!(wake, body)
        prescribe_wake!(wake, Gam, info)
        nothing
    end
end

freeze_wake(wake) = (nodes=[copy(n) for n in wake.nodes],
                     strength=[copy(s) for s in wake.strength],
                     nwakes=wake.nwakes[])

function restore_wake!(wake, fz)
    for (n, cn) in zip(wake.nodes, fz.nodes)
        n .= cn
    end
    for (s, cs) in zip(wake.strength, fz.strength)
        s .= cs
    end
    wake.nwakes[] = fz.nwakes
    return wake
end

frozen_maneuver(wake, fz) = (frames, systems, wakes, t) -> (restore_wake!(wake, fz); nothing)

# ---------------- telemetry ----------------------------------------------------
mutable struct RunStore
    gammas::Vector{Vector{Float64}}
    deltas::Vector{Float64}
    gr::Vector{Any}
    cap::Any
end
RunStore() = RunStore(Vector{Float64}[], Float64[], [], nothing)

function make_callback(body, wake, info, store::RunStore; gr::Bool, capture_at::Int)
    return (nt) -> begin
        st = nt.formulation_state
        G = gamma_of(body.strength, info)
        delta = isempty(store.gammas) ? NaN :
            maximum(abs.(G .- store.gammas[end])) / max(maximum(abs.(G)), eps())
        push!(store.gammas, G)
        push!(store.deltas, delta)
        all(isfinite, body.strength) || error("non-finite strengths at step $(nt.i_step)")
        if gr
            push!(store.gr, (i=nt.i_step, q=copy(st.green.q), sigma=copy(st.sigma),
                lambda=pnl._green_lambda(st.green)))
        end
        if nt.i_step == capture_at
            store.cap = (strength=copy(body.strength), wake=freeze_wake(wake))
        end
        nothing
    end
end

# ---------------- metrics (stage-1 locked + supersession additions) ------------
align(q, a) = q .- dot(a, q) / sum(a)
rmsA(q, a) = sqrt(sum(a .* q .^ 2) / sum(a))

"Raw M1, aligned M1a, E_inf (all / TE-excluded, aligned), offset M1b."
function mu_metrics(muGR, muD, a, temask)
    dscale = rmsA(muD, a)
    E_raw = rmsA(muGR .- muD, a) / dscale
    mGRa, mDa = align(muGR, a), align(muD, a)
    ascale = rmsA(mDa, a)
    E_al = rmsA(mGRa .- mDa, a) / ascale
    d = mGRa .- mDa
    E_inf_all = maximum(abs.(d)) / ascale
    keep = .!temask
    E_inf_ex = maximum(abs.(d[keep])) / ascale
    offset = dot(a, muD .- muGR) / sum(a)
    return (; E_raw, E_al, E_inf_all, E_inf_ex, offset)
end

"M2: same body/stations, dy-weighted rms and max, /D-scale; plus dGtot, dCL."
function gamma_metrics(GGR, GD, info)
    scale = sqrt(sum(info.dy .* GD .^ 2) / sum(info.dy))
    grms = sqrt(sum(info.dy .* (GGR .- GD) .^ 2) / sum(info.dy)) / max(scale, eps())
    gmax = maximum(abs.(GGR .- GD)) / max(maximum(abs.(GD)), eps())
    return (; grms, gmax,
        dGtot=gtot_of(GGR, info) - gtot_of(GD, info),
        dCL=cl_of(GGR, info) - cl_of(GD, info))
end

# ---------------- runs ----------------------------------------------------------
"Phase-A-style fixed point with the given route; returns converged record."
function run_fixedpoint(route, lev)
    runid = "SC-$(route.id)-L$lev"
    body = make_capped_wing(lev)
    info = shed_info(body)
    frames = pitching_wing_frame(body, PIVOT, deg2rad(AOA))
    set_wake_Das!(body, VINF; magnitude=DAS)
    wake = pnl.PanelWake(body; nwakerows=NFREE, include_final_filament=false)
    pnl.update_TE!(wake, body)
    prescribe_wake!(wake, zeros(length(info.y)), info)
    man = fixedpoint_maneuver(body, wake, info)
    store = RunStore()
    cb = make_callback(body, wake, info, store; gr=route.gr, capture_at=NITER_A - 1)
    trange = range(0.0, step=DT, length=NITER_A)
    t_el = @elapsed pnl.simulate!((body,), (wake,), frames, man, Uinf_f, trange;
        body_solvers=(pnl.Backslash(body),), backend=DIRECT, path=nothing,
        formulation=route.form(), step_telemetry_callback=cb, verbose=false)
    cap = store.cap
    cap === nothing && error("$runid: capture step never fired")
    deltaA = store.deltas[end]
    G = gamma_of(cap.strength, info)
    kx = kutta_xcheck(body, info, cap.strength)
    @printf("  %-10s N=%-6d Gtot=%+.6e CL=%+.5f dA=%.1e kx=%.1e t=%.0fs\n",
        runid, body.ncells, gtot_of(G, info), cl_of(G, info), deltaA, kx, t_el)
    flush(stdout)
    return (; runid, body, info, store, cap, G, deltaA, kutta_x=kx,
        N=body.ncells, t_el, converged=(isfinite(deltaA) && deltaA <= TOL_A))
end

"Single GR solve on a frozen wake (fresh body); returns strengths + GR state."
function run_frozen_gr(lev, fz)
    runid = "FZ-GR-L$lev"
    body = make_capped_wing(lev)
    info = shed_info(body)
    frames = pitching_wing_frame(body, PIVOT, deg2rad(AOA))
    set_wake_Das!(body, VINF; magnitude=DAS)
    wake = pnl.PanelWake(body; nwakerows=NFREE, include_final_filament=false)
    restore_wake!(wake, fz)
    man = frozen_maneuver(wake, fz)
    store = RunStore()
    # length-2 trange: FLOWPanel's stepper indexes the previous time sample, so a
    # single-sample range is out of bounds; the maneuver re-freezes the wake every
    # step, so both solves see the identical frozen wake (capture the last).
    cb = make_callback(body, wake, info, store; gr=true, capture_at=1)
    t_el = @elapsed pnl.simulate!((body,), (wake,), frames, man, Uinf_f,
        range(0.0, step=DT, length=2);
        body_solvers=(pnl.Backslash(body),), backend=DIRECT, path=nothing,
        formulation=pnl.GreenReconstruction(gauge=:area_mean),
        step_telemetry_callback=cb, verbose=false)
    cap = store.cap
    cap === nothing && error("$runid: capture step never fired")
    G = gamma_of(cap.strength, info)
    @printf("  %-10s N=%-6d Gtot=%+.6e CL=%+.5f t=%.0fs\n",
        runid, body.ncells, gtot_of(G, info), cl_of(G, info), t_el)
    flush(stdout)
    return (; runid, body, info, store, cap, G, wake, N=body.ncells, t_el)
end

"GR-system diagnostics (residual, gauge defect) at the measurement solve."
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

# ---------------- main ----------------------------------------------------------
lines = String[]
say(s) = (push!(lines, s); println(s); flush(stdout))

println("== 052e.3 Phase 1 $(SMOKE ? "(SMOKE, unregistered)" : "(REGISTERED)") ==")
println("levels=$(LEVELS) NFREE=$NFREE DT=$DT NITER_A=$NITER_A TOL_A=$TOL_A")
println("datadir=$DATADIR threads=$(Threads.nthreads())")
flush(stdout)

results = []
try
    for lev in 1:length(LEVELS)
        println("---- level L$lev $(LEVELS[lev]) ----"); flush(stdout)
        recD = run_fixedpoint(ROUTES[1], lev)
        recGR = run_fixedpoint(ROUTES[2], lev)
        recFZ = run_frozen_gr(lev, recD.cap.wake)

        recD.converged || say("WARN SC-D-L$lev fixed point delta=$(recD.deltaA) > $TOL_A")
        recGR.converged || say("WARN SC-GR-L$lev fixed point delta=$(recGR.deltaA) > $TOL_A")

        a = pnl._panel_areas(recD.body)
        temask = te_mask(recD.body, recD.info)
        muD = recD.cap.strength[:, end]

        # M1b prediction: area-mean of the direct wake potential on the frozen wake
        qf = zeros(length(a))
        pnl._wake_potential!(qf, recFZ.body, (recFZ.wake,), DIRECT)
        qf_mean = dot(a, qf) / sum(a)

        sc_mu = mu_metrics(recGR.cap.strength[:, end], muD, a, temask)
        fz_mu = mu_metrics(recFZ.cap.strength[:, end], muD, a, temask)
        sc_g = gamma_metrics(recGR.G, recD.G, recD.info)
        fz_g = gamma_metrics(recFZ.G, recD.G, recD.info)
        gdSC = green_diag(recGR.body, recGR.store.gr[end])
        gdFZ = green_diag(recFZ.body, recFZ.store.gr[end])

        # per-level CSVs
        open(joinpath(DATADIR, "mu_L$lev.csv"), "w") do io
            println(io, "area,te_adjacent,mu_D,mu_GR_sc,mu_GR_fz,q_f")
            muS = recGR.cap.strength[:, end]; muF = recFZ.cap.strength[:, end]
            for i in eachindex(a)
                @printf(io, "%.9e,%d,%.9e,%.9e,%.9e,%.9e\n",
                    a[i], temask[i], muD[i], muS[i], muF[i], qf[i])
            end
        end
        open(joinpath(DATADIR, "gamma_L$lev.csv"), "w") do io
            println(io, "y,dy,Gamma_D,Gamma_GR_sc,Gamma_GR_fz")
            for k in eachindex(recD.G)
                @printf(io, "%.9e,%.9e,%.9e,%.9e,%.9e\n", recD.info.y[k],
                    recD.info.dy[k], recD.G[k], recGR.G[k], recFZ.G[k])
            end
        end

        say(@sprintf("L%d N=%-6d  M1 raw: sc=%.4e fz=%.4e   M1a aligned: sc=%.4e fz=%.4e",
            lev, recD.N, sc_mu.E_raw, fz_mu.E_raw, sc_mu.E_al, fz_mu.E_al))
        say(@sprintf("L%d   Einf(al) sc=(%.3e,%.3e ex) fz=(%.3e,%.3e ex)",
            lev, sc_mu.E_inf_all, sc_mu.E_inf_ex, fz_mu.E_inf_all, fz_mu.E_inf_ex))
        say(@sprintf("L%d   M1b offset: sc=%+.6e fz=%+.6e  pred |<q_f>_A|=%.6e",
            lev, sc_mu.offset, fz_mu.offset, abs(qf_mean)))
        say(@sprintf("L%d   M2 Gamma: sc grms=%.4e gmax=%.4e dCL=%+.5f | fz grms=%.4e gmax=%.4e dCL=%+.5f",
            lev, sc_g.grms, sc_g.gmax, sc_g.dCL, fz_g.grms, fz_g.gmax, fz_g.dCL))
        say(@sprintf("L%d   GR diag: sc(res=%.2e gd=%.2e lam=%+.2e) fz(res=%.2e gd=%.2e lam=%+.2e)",
            lev, gdSC.res, gdSC.gd, gdSC.lambda, gdFZ.res, gdFZ.gd, gdFZ.lambda))
        say(@sprintf("L%d   scalars: CL_D=%+.5f CL_GR_sc=%+.5f CL_GR_fz=%+.5f dA=(%.1e,%.1e) kx=(%.1e,%.1e)",
            lev, cl_of(recD.G, recD.info), cl_of(recGR.G, recD.info),
            cl_of(recFZ.G, recD.info), recD.deltaA, recGR.deltaA,
            recD.kutta_x, recGR.kutta_x))

        push!(results, (; lev, N=recD.N, sc_mu, fz_mu, sc_g, fz_g, qf_mean))
        GC.gc()
    end
finally
    # silo copies have no .git — fall back to the rsync-time provenance file
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
        println(io, "052e.3 Phase 1 $(SMOKE ? "SMOKE (UNREGISTERED — values meaningless)" : "REGISTERED")")
        println(io, "date=$(chomp(read(`date +%Y-%m-%d`, String))) julia=$(VERSION) threads=$(Threads.nthreads()) script_sha256=$script_sha")
        println(io, "FLOWPanel: $(gitinfo(joinpath(HERE, "..", "..", "..", "FLOWPanel.jl")))")
        println(io, "FastMultipole: $(gitinfo(joinpath(HERE, "..", "..")))")
        if isfile(provfile)
            println(io, "silo_provenance:")
            foreach(l -> println(io, "  " * l), eachline(provfile))
        end
        println(io, "fixture: AOA=$(AOA)deg |U|=$UMAG levels=$(LEVELS) NFREE=$NFREE ROWLEN=$ROWLEN DT=$DT NITER_A=$NITER_A TOL_A=$TOL_A")
        println(io, "routes: D=DirectWakePotential GR=GreenReconstruction(:area_mean) | comparisons: SC self-consistent fixed points, FZ GR-on-frozen-D-wake (supersession H1)")
        println(io, "reporting: no pass/fail gates (prereg) — raw values below, Ryan rules")
        for l in lines
            println(io, l)
        end
    end
    println("\n===== 052e.3 PHASE 1 $(SMOKE ? "SMOKE" : "MEASUREMENTS") =====")
    foreach(println, lines)
    println("outputs: $DATADIR")
    println("=========================================")
end
