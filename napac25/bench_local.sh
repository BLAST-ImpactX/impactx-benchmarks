#!/usr/bin/env bash
#
# Run all benchmarks on the local machine.
# Save the logs.
#

set -eu -o pipefail

DATESTR=$(date +"%Y-%m-%d_%H-%M-%S")

# log GPU(s) on the machine, if any
nvidia-smi 2>&1 | tee bench_logs/${DATESTR}_$(hostname).log || true

./bench_scenarios.py 2>&1 | tee -a bench_logs/${DATESTR}_$(hostname).log
