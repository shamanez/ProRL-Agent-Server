#!/bin/bash
# Stage 0 baseline entry point.
#
# We run the trainer inside the verlai/verl docker image because the host
# env (/opt/pytorch) ships vllm 0.19 but verl 60138ebd requires vllm 0.8.x.
# See plans-n-solutions/solutions/stage0_baseline.md §Problem 6 for the
# decision trail.
#
# Keep this file as the public entry point; the docker details live in
# s0_baseline_docker.sh so the tmux invocation is short.
exec "$(dirname "$0")/s0_baseline_docker.sh" "$@"
