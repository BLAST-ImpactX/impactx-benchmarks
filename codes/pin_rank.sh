#!/usr/bin/env bash
# Pin one MPI rank to its slice of the core-budget CPU list, then exec the command.
# usage: pin_rank.sh <threads_per_rank> <cpu_csv> <command...>
# Rank index comes from MPICH (PMI_RANK); each rank takes `threads` CPUs from the list.
set -e
threads="$1"; cpus="$2"; shift 2
rank="${PMI_RANK:-${PMI_ID:-${OMPI_COMM_WORLD_RANK:-0}}}"
IFS=',' read -ra arr <<< "$cpus"
start=$(( rank * threads )); mine=""
for ((i=0; i<threads; i++)); do
  j=$(( start + i ))
  if [ "$j" -lt "${#arr[@]}" ]; then mine="${mine:+$mine,}${arr[$j]}"; fi
done
exec taskset -c "$mine" "$@"
