#!/usr/bin/env julia
# Auto-generated benchmark run script: SciBmad.jl / fodo_exact.
# MatrixKick = a symplectic kick-MATRIX-kick splitting (Yoshida order 4), the genuine EXACT
# NON-PARAXIAL quad map. Verified from BeamTracking v0.7.0 source (src/kernels/quadrupole_kick.jl,
# https://github.com/bmad-sim/BeamTracking.jl/blob/v0.7.0/src/kernels/quadrupole_kick.jl):
# the "matrix" piece is a linear chromatic sin/cos map with kappa=sqrt(k1/(1+delta)) (paraxial by
# itself), BUT the surrounding "kick" pieces apply a position kick using the EXACT longitudinal
# momentum Ps=sqrt((1+delta)^2-px^2-py^2) -- i.e. s*px*(px^2+py^2)/(P*Ps*(P+Ps)), cubic+ in the
# transverse angles. So the NET map is non-paraxial (amplitude-dependent focusing / spherical
# aberration; does not exactly preserve emittance), converging to the exact quad Hamiltonian.
# Empirically it clusters with ImpactX ExactQuad (-0.11% on this scenario), NOT with the paraxial
# pole. The Drift is SciBmad's exact non-paraxial map -> a fully exact (non-paraxial) FODO model.
using SciBmad
using Random, Statistics

const NPART = 1000
mass_MeV = 0.51099895069
kin_energy_MeV = 100.0
emit_x = 0.0001; emit_y = 0.0001
beta_x = 0.01; beta_y = 0.01
alpha_x = 0.0; alpha_y = 0.0
sigma_t = 0.001; sigma_p = 0.01
quad_length = 0.5; drift_length = 0.1
k1 = 2.0
total_energy_eV = (kin_energy_MeV + mass_MeV) * 1e6

Random.seed!(12345)
function gaussian_plane(emit, beta, alpha, n)
    s11 = emit * beta; s12 = -emit * alpha; s22 = emit * (1 + alpha^2) / beta
    L11 = sqrt(s11); L21 = s12 / L11; L22 = sqrt(max(s22 - L21^2, 0.0))
    z1 = randn(n); z2 = randn(n)
    return L11 .* z1, L21 .* z1 .+ L22 .* z2
end

v0 = zeros(Float64, NPART, 6)  # SP/DP via cfg.precision
x, px = gaussian_plane(emit_x, beta_x, alpha_x, NPART)
y, py = gaussian_plane(emit_y, beta_y, alpha_y, NPART)
v0[:, 1] .= x; v0[:, 2] .= px; v0[:, 3] .= y; v0[:, 4] .= py
v0[:, 5] .= sigma_t .* randn(NPART); v0[:, 6] .= sigma_p .* randn(NPART)

sp = Species("electron")
# MatrixKick = exact NON-PARAXIAL quad (kick-matrix-kick w/ exact-Ps drift correction; Yoshida o4)
qf = Quadrupole(Kn1=k1, L=quad_length, tracking_method=MatrixKick())
qd = Quadrupole(Kn1=-k1, L=quad_length, tracking_method=MatrixKick())
d = Drift(L=drift_length)
bl = Beamline([qf, d, qd, d], species_ref=sp, E_ref=total_energy_eV)

function emit_geom(u, up)
    u = u .- mean(u); up = up .- mean(up)
    return sqrt(max(mean(u .* u) * mean(up .* up) - mean(u .* up)^2, 0.0))
end

function track_fresh()
    b = Bunch(copy(v0); species=sp, p_over_q_ref=bl.p_over_q_ref)
    track!(b, bl; use_cpu_multithreading=true, use_KA=false, use_explicit_SIMD=false)
    return b
end

track_fresh()  # warm-up: Julia JIT compilation (NOT timed)
b = Bunch(copy(v0); species=sp, p_over_q_ref=bl.p_over_q_ref)
t0 = time_ns()
track!(b, bl; use_cpu_multithreading=true, use_KA=false, use_explicit_SIMD=false)
dt = time_ns() - t0

vf = Array(b.v)
sx = std(vf[:, 1]); sy = std(vf[:, 3])
ex = emit_geom(vf[:, 1], vf[:, 2]); ey = emit_geom(vf[:, 3], vf[:, 4])
println("Track: ", dt, "ns")
println("Validate: {\"sigma_x\": ", sx, ", \"sigma_y\": ", sy,
        ", \"emit_x\": ", ex, ", \"emit_y\": ", ey, "}")
