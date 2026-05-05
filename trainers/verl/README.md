# VERL — TrainerAdapter

VERL FSDP implementation of the `TrainerAdapter` protocol (GRPO/DAPO).

**Install** (inside Docker container only — never on host):
```bash
pip install --no-deps -e /workspace/trainers/verl
```

**Start**:
```bash
bash trainers/verl/scripts/start.sh
```

Connects only to LiveStore and PolicyRegistry (BC-15). No parquet path, no ProRL URL, no vLLM URL.

See root `CLAUDE.md` for the full startup sequence and boundary conditions.
