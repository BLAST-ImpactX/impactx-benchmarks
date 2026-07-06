# Instantiate the SciBmad GPU Julia project: same SciBmad checkout as the CPU project,
# PLUS CUDA.jl (the KernelAbstractions CUDA backend BeamTracking dispatches to with
# use_KA=true on a CuArray bunch). Run via:
#   julia --project=codes/scibmad-gpu codes/scibmad-gpu/build.jl
using Pkg

# Use the local SciBmad.jl checkout if present (override path with SCIBMAD_PATH); else
# add the registered package (e.g. in CI). BeamTracking.jl / Beamlines.jl resolve from
# the General registry as normal deps either way. Kept identical to the CPU build so the
# GPU and CPU runs track the SAME SciBmad source (only the device backend differs).
scibmad_path = get(ENV, "SCIBMAD_PATH", "/home/axel/src/SciBmad.jl")
if isdir(scibmad_path)
    Pkg.develop(path=scibmad_path)
else
    Pkg.add("SciBmad")
end

Pkg.add("CUDA")     # GPU backend (CuArray + KernelAbstractions CUDA dispatch)

Pkg.instantiate()
Pkg.precompile()
println("SciBmad GPU project instantiated from ", scibmad_path, " (+ CUDA.jl)")
