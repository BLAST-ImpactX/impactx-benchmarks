#!/usr/bin/env bash
#
# Run all benchmarks on the local machine.
# Save the logs.
#

DATESTR=$(date +"%Y-%m-%d_%H-%M-%S")

./bench_scenarios.py 2>&1 | tee bench_logs/${DATESTR}_$(hostname).log
