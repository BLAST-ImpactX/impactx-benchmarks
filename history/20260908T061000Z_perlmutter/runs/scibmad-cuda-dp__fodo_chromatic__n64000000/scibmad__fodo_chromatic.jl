#!/usr/bin/env julia
# Auto-generated benchmark run script: SciBmad.jl / fodo_chromatic.
# Quad MatrixKick is chromatic-paraxial, but SciBmad's drift is exact-only (no paraxial
# drift), so the drift runs the costlier EXACT map -> marked with an asterisk (untuned).
# Exact thick CHROMATIC quad: MatrixKick = analytic sin/cos map with k1/(1+delta)
# (the same model as ImpactX ChrQuad / Cheetah drift_kick_drift / pyAT QuadLinearPass /
# Xsuite mat-kick-mat), plus the exact non-paraxial drift.
using SciBmad
using Random, Statistics
using CUDA  # GPU: CuArray bunch + KernelAbstractions CUDA backend

const NPART = 64000000
mass_MeV = 0.51099895069
kin_energy_MeV = 100.0
emit_x = 1e-09; emit_y = 1e-09
beta_x = 1.0; beta_y = 1.0
alpha_x = 0.0; alpha_y = 0.0
sigma_t = 0.001; sigma_p = 0.01
quad_length = 0.1; drift_length = 0.5
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

v0 = CuArray(v0)  # move the (CPU-seeded) beam to the GPU
sp = Species("electron")
# MatrixKick = exact thick chromatic quad (matrix-kick-matrix, k1/(1+delta))
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
    CUDA.@sync track!(b, bl; use_KA=true, use_explicit_SIMD=false)
    return b
end

track_fresh()  # warm-up: Julia JIT compilation (NOT timed)
b = Bunch(copy(v0); species=sp, p_over_q_ref=bl.p_over_q_ref)
t0 = time_ns()
CUDA.@sync track!(b, bl; use_KA=true, use_explicit_SIMD=false)
dt = time_ns() - t0

vf = Array(b.v)
sx = std(vf[:, 1]); sy = std(vf[:, 3])
ex = emit_geom(vf[:, 1], vf[:, 2]); ey = emit_geom(vf[:, 3], vf[:, 4])
println("Track: ", dt, "ns")
println("Validate: {\"sigma_x\": ", sx, ", \"sigma_y\": ", sy,
        ", \"emit_x\": ", ex, ", \"emit_y\": ", ey, "}")
