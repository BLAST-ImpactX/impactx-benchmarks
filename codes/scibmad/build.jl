# Instantiate the SciBmad Julia project from the local source checkout and precompile.
# Run via:  julia --project=codes/scibmad codes/scibmad/build.jl
using Pkg

# Use the local SciBmad.jl checkout if present (override path with SCIBMAD_PATH); else
# add the registered package (e.g. in CI). BeamTracking.jl / Beamlines.jl resolve from
# the General registry as normal deps either way.
scibmad_path = get(ENV, "SCIBMAD_PATH", "/home/axel/src/SciBmad.jl")
if isdir(scibmad_path)
    Pkg.develop(path=scibmad_path)
else
    Pkg.add("SciBmad")
end

Pkg.instantiate()
Pkg.precompile()
println("SciBmad project instantiated from ", scibmad_path)
