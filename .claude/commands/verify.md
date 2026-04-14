---
description: Full verification pass — lint, tests, type-check. Delegates to the verification-loop skill.
---

# /verify — Verification Loop

Apply the `verification-loop` skill (`.claude/skills/verification-loop.md`) at the depth appropriate for the current change.

## Arguments

`$ARGUMENTS`  (optional: `fast`, `full`, or `ci`)

## Project-specific invocations for this repo

**Fast (default, ~30-90s):**
```bash
make lint                                                        # pre-commit: ruff + mypy + pyproject-fmt
pytest -m "not integration and not slow and not real_data" tests/ -q
```

**Full (includes integration, ~5-15min):**
```bash
make lint
pytest -m "not slow" tests/
```

**CI parity (everything, long):**
```bash
make lint
make lint-scripts
pytest tests/
```

## Reporting

Report only verdicts and blockers — do not narrate every step. If lint fails, surface the exact ruff/mypy errors. If tests fail, surface the first 3 failures and the test command to reproduce each one locally.

## Related

- Skill: `skills/verification-loop.md`
- Command: `/test-coverage` for coverage detail
- Command: `/build-fix` if verification surfaces build errors
