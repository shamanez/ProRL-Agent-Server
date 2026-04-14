---
description: Test-driven development workflow. Delegates to the tdd-workflow skill.
---

# /tdd — Test-Driven Development

Apply the `tdd-workflow` skill (`.claude/skills/tdd-workflow.md`). Stay strict on RED → GREEN → REFACTOR.

## Arguments

`$ARGUMENTS`

## Project-specific commands for this repo

Fast test loop (use during tight iteration):
```bash
pytest -m "not integration and not slow and not real_data" tests/ -q
```

Unit-only run with coverage on a specific module:
```bash
pytest --cov=<module> --cov-report=term-missing tests/<module>/
```

Single test:
```bash
pytest tests/<path>/test_foo.py::test_case -v
```

Pytest markers available in this repo: `asyncio`, `integration`, `real_data`, `slow`. Skip integration/slow/real_data during RED→GREEN cycles; include them before declaring the feature done.

Environment for runtime tests (set if you touch `tests/runtime/**`):
```bash
export TEST_RUNTIME=singularity
export RUN_AS_OPENHANDS=False
export PYTHONPATH=.
```

## Related

- Skill: `skills/tdd-workflow.md`
- Agent: `agents/tdd-guide.md`
- After green: run `/verify` for full verification before committing.
