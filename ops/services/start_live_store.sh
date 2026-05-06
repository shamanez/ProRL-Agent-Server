#!/usr/bin/env bash
# Start the LiveStore gRPC service (slot 5.4 — S1).
# Run BEFORE start_rollout_manager.sh and BEFORE the trainer.
#
# Step 3a in the startup sequence (can run parallel with start_policy_registry.sh).
# Health: GET the socket existence + python health check.
#
# Env vars:
#   LIVE_STORE_SOCKET    (default /tmp/prorl_live_store.sock)
#   LIVE_STORE_MAX_SIZE  (default 256 groups)
#   STALENESS_CUTOFF_K   (default 32 policy versions)
#   NO_PROGRESS_TIMEOUT  (default 1800 seconds)
#
# BC-16: trainer blocks server-side in get_batch until buffer is warm.
# The no-progress timeout (1800s) is the safety abort, not a short RPC timeout.
#
# Staleness semantics: groups are stamped created_at_step=policy_version at
# dispatch time (loop.py). A group is stale when the live policy has advanced
# more than k versions since the group was collected. k=32 means trajectories
# collected under a policy >32 gradient updates old are rejected — keeping
# training approximately on-policy for GRPO/DAPO.
# Restart safety: after resume from step N the manifest shows version=N, so
# new groups get created_at_step=N and trainer's get_batch(current_step=N+1)
# sees age=1 ≤ 32 → fresh. Old pre-restart groups with small created_at_step
# are naturally evicted when N-created_at_step > 32.
set -euo pipefail

SOCKET="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}"
MAX_SIZE="${LIVE_STORE_MAX_SIZE:-256}"
K="${STALENESS_CUTOFF_K:-32}"
NO_PROGRESS="${NO_PROGRESS_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[live_store] starting on unix:${SOCKET} max_size=${MAX_SIZE} k=${K}"

# ROLLOUT_FABRIC_PYTHON: override to use a fabric-only venv (see docs/service-envs.md).
# POETRY_PYTHON: set to the output of: poetry env info --path)/bin/python
# Default fallback: the pre-populated env on this machine.
_DEFAULT_PYTHON="$(cd "${REPO_ROOT}" && poetry env info --path 2>/dev/null)/bin/python"
PYTHON="${ROLLOUT_FABRIC_PYTHON:-${POETRY_PYTHON:-${_DEFAULT_PYTHON}}}"

exec "${PYTHON}" -c "
import logging, os, signal, sys, time
logging.basicConfig(level=os.environ.get('LOG_LEVEL','INFO'),
    format='%(asctime)s %(levelname)s live_store: %(message)s')
sys.path.insert(0, '${REPO_ROOT}/core')
from rollout_fabric.live_store.server import serve
server = serve(
    socket_path='${SOCKET}',
    max_size=${MAX_SIZE},
    staleness_cutoff_k=${K},
    no_progress_timeout_s=${NO_PROGRESS},
)
print('[live_store] healthy unix:${SOCKET}', flush=True)
def _stop(s, f):
    server.stop(grace=5.0)
    sys.exit(0)
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
"
