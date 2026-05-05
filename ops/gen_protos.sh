#!/usr/bin/env bash
# Regenerate gRPC stubs from schemas/proto/*.proto into schemas/_gen/.
#
# Generated files are CHECKED IN — the trainer/worker/store hosts only
# need grpcio + protobuf at runtime, never grpcio-tools. Run this only
# when a .proto changes.
#
# Run from the repo root:  bash scripts/gen_protos.sh
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p core/rollout_fabric/schemas/_gen
touch core/rollout_fabric/schemas/_gen/__init__.py

poetry run python -m grpc_tools.protoc \
  -I=core/rollout_fabric/schemas/proto \
  --python_out=core/rollout_fabric/schemas/_gen \
  --grpc_python_out=core/rollout_fabric/schemas/_gen \
  core/rollout_fabric/schemas/proto/live_store.proto \
  core/rollout_fabric/schemas/proto/policy_registry.proto

# Rewrite generated imports to use the package-qualified path. protoc
# emits ``import live_store_pb2`` which fails when ``schemas/_gen`` is
# not on sys.path; rewriting to ``from rollout_fabric.schemas._gen import ...`` works
# regardless of how the consumer launches.
for f in core/rollout_fabric/schemas/_gen/live_store_pb2_grpc.py schemas/_gen/policy_registry_pb2_grpc.py; do
  if [ -f "$f" ]; then
    sed -i 's/^import \(.*_pb2\)$/from rollout_fabric.schemas._gen import \1/' "$f"
  fi
done

echo "Stubs regenerated in schemas/_gen/"
