#!/bin/bash
# Thin symmetry wrapper so operators don't have to remember the subcommand.
# Kills the remote pool and rsyncs child logs back to /tmp (see
# launch_remote_vllm_pool.sh::do_stop for why the log copy is here).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/launch_remote_vllm_pool.sh" stop
