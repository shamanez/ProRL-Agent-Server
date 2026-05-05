#!/usr/bin/env bash
# TrainerAdapter launcher for slime (stub — not yet implemented).
# See trainer_adapters/slime/README.md for integration steps.
#
# When implemented, this script should follow the same pattern as start_trainer_verl.sh:
#   docker run {slime_image} -v $(pwd):/workspace \
#     -e LIVE_STORE_SOCKET=/tmp/prorl_live_store.sock \
#     -e REGISTRY_SOCKET=/tmp/prorl_policy_registry.sock \
#     bash -c "pip install -e /workspace/trainer_integration/slime && python -m slime.train ..."
#
# Key constraints (BC-14/15):
#   - The trainer must NOT receive a parquet path.
#   - The trainer must NOT receive a ProRL URL or vLLM URL.
#   - Rollout data comes exclusively from LiveStoreClient.get_batch().
set -euo pipefail

echo "[start_trainer_slime] ERROR: slime TrainerAdapter not yet implemented."
echo "  See trainer_adapters/slime/README.md for integration steps."
exit 1
