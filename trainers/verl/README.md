# VERL — TrainerAdapter

VERL FSDP implementation of the `TrainerAdapter` protocol (GRPO/DAPO). One of several
possible trainers — slime and ROLL are under `trainers/`. Swap by changing the Docker
image and Hydra launch script; the fabric contracts stay the same.

## Contract (BC-15)

Connects **only** to LiveStore (`get_batch`) and PolicyRegistry (`publish_policy_version`).
No parquet path, no ProRL URL, no vLLM URL. Hard-aborts if any vLLM endpoint fails to
reload the LoRA (BC-9).

## How VERL is installed (not baked into the image)

```
Host /tmp/verl  ──bind-mount──►  /opt/verl  (inside container)
                                      │
                              pip install --no-deps -e /opt/verl        ← upstream VERL
                              pip install --no-deps -e /workspace/trainers/verl  ← our patch
```

VERL is always installed from the host's `/tmp/verl` checkout at container start.
Update VERL by changing `/tmp/verl` on the host — no image rebuild needed. The
`verl_custom` patch package adds the LiveStore consumer seam and PolicyRegistry
publish hook on top of the pinned upstream.

Pinned commit: `a4351480` — see `trainers/verl/VERL_PIN.md`. Do not touch `/tmp/verl`.

## Start

```bash
bash trainers/verl/scripts/start.sh
```

Key env knobs: `BATCH_SIZE`, `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `STALENESS_CUTOFF_K`,
`SWAP_PROTOCOL`. Full knob table: `docs/TRAINING_OPERATIONS.md` Section 8.
