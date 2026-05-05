# ROLL TrainerAdapter (stub)

**Upstream:** https://github.com/alibaba/ROLL

This is a documentation stub for a future ROLL-based TrainerAdapter.

## Integration steps

1. Add `trainer_adapters/roll/pad.py` implementing `pack_unpadded_groups()`.
   Reference the VERL implementation in `../verl/pad.py`.

2. Create `trainer_integration/roll/` with a `pyproject.toml` + integration layer.

3. Create `scripts/adapters/start_trainer_roll.sh` using the ROLL Docker image.

See `../README.md` for the step-by-step guide.
See `../../schemas/protocols/PLUGGING_IN.md` for the Protocol contract.
