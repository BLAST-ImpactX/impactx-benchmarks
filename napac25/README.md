## NAPAC25 Benchmarks

These are benchmarks for new features (e.g., SIMD) in ImpactX and ImpactX/Cheetah (C++/PyTorch) comparisons.

The entry point is the script `./bench_scenarios.py`.

You can call this script and log the output via:
- `./bench_local.sh`: run on your local machine
- `sbatch bench_perlmutter.sbatch`: benchmark on an exclusive Perlmutter node

For Perlmutter, I added the following conda configs to avoid running out of HOME disk space for conda envs:

```
conda config --add envs_dirs $PSCRATCH/conda/envs
conda config --add pkgs_dirs $PSCRATCH/conda/pkgs
```
(options become part of `~/.condarc`)

Commit the `timings.yaml` file after each benchmark was run.
