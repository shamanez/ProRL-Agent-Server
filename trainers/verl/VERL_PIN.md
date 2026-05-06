# VERL Pin

## Pinned commit

```
a4351480871347092436d17573ad3ccf75b24122
[misc] fix: add missing __init__.py files to package directories (#5209)
https://github.com/verl-project/verl
```

## Why this commit

- VERL 0.8.0.dev — first stable v0.8 series that ships `engine_workers.py`
  (the replacement for the removed `AsyncActorRolloutRefWorker`).
- Our `fsdp_workers.py` shim targets this API: it falls back from
  `verl.workers.fsdp_workers` → `verl.workers.engine_workers` and patches
  `init_model` / `update_weights` for the LiveStore-only path.
- `get_per_tensor_param(base_sync_done=True)` is available at this commit
  and correctly returns only LoRA A/B matrices via `get_peft_model_state_dict()`
  rather than the full base-model weights.

## Key API changes vs v0.4 (relevant to our patches)

| v0.4 | v0.8 |
|------|------|
| `verl.workers.fsdp_workers.AsyncActorRolloutRefWorker` | removed; `verl.workers.engine_workers.ActorRolloutRefWorker` |
| `omega_conf_to_dataclass(config.actor)` accepted extra fields | raises on unknown fields — patched via `_ConfigProxy` |
| `get_per_tensor_param()` always returned full params | `get_per_tensor_param(base_sync_done=True)` returns only LoRA delta |
| `verl.workers.config` was a single file | split into `verl/workers/config/` package; re-exports via `__init__` |

## To update the pin

1. Clone the new commit to `/tmp/verl` and test the import compat check
   (see `trainers/verl/scripts/start.sh` inline check block).
2. Update `VERL_COMMIT` in `start.sh`.
3. Update this file with the new commit SHA + one-line summary.
4. Run `make lint` and the fast test loop before committing.

## Auto-download

`trainers/verl/scripts/start.sh` clones this commit automatically if
`/tmp/verl` does not exist. If `/tmp/verl` exists but is at a different
SHA, it prints a warning and continues with whatever is there.

To force a fresh clone:

```bash
rm -rf /tmp/verl
bash trainers/verl/scripts/start.sh
```
