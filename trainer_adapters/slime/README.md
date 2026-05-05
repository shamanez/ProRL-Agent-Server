# slime TrainerAdapter (stub)

**Upstream:** https://github.com/THUDM/slime

This is a documentation stub for a future slime-based TrainerAdapter.

## Integration steps

1. Add `trainer_adapters/slime/pad.py` implementing `pack_unpadded_groups()`.
   Reference the VERL implementation in `../verl/pad.py`.

2. Create `trainer_integration/slime/` with a `pyproject.toml` + integration layer.

3. Create `scripts/adapters/start_trainer_slime.sh` using the slime Docker image.

See `../README.md` for the step-by-step guide.
See `../../schemas/protocols/PLUGGING_IN.md` for the Protocol contract.
