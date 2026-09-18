# Subprocess helper for fgs_chunked_test.jl's cross-process thread-count
# invariance check. Rebuilds the exact fixture of the parent test's
# `cold_fixed_solve!` reference and writes the fixed-iteration residual and
# strength bit patterns (UInt64, space-separated, two lines) to ARGS[1].
# Run as: julia -t N --project=<test env> fgs_chunked_threadcheck.jl <outfile>

import FastMultipole
using FastMultipole

include(joinpath(@__DIR__, "gravitational.jl"))

# Gravitational <-> solver compatibility overloads (duplicated from the top of
# solve_test.jl, which is not includable here without running its testsets)
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

system = generate_gravitational(20260918, 800)
direct!(system; scalar_potential=true, gradient=false)
system.potential[1, :] .*= -1.0

fgs = FastMultipole.FastGaussSeidel((system,), (system,);
    expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
    shrink=true, recenter=false, sweep_order=:chunked, chunks=4)

for i in eachindex(system.bodies)
    body = system.bodies[i]
    system.bodies[i] = typeof(body)(body.position, body.radius, 0.0)
end
residuals = Float64[]
FastMultipole.solve!(system, fgs; scalar_potential=true, gradient=false,
    max_iterations=6, inner_iterations=2, tolerance=-1.0,
    reverse_pass=false, final_update=false, verbose=false,
    callback=(_, residual) -> push!(residuals, residual))
strengths = [body.strength for body in system.bodies]

open(ARGS[1], "w") do io
    println(io, join(collect(reinterpret(UInt64, residuals)), ' '))
    println(io, join(collect(reinterpret(UInt64, strengths)), ' '))
end
