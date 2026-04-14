---
name: testing-conventions
description: Pytest conventions — markers, fixtures, env vars. Loaded only when editing tests/.
globs: ["tests/**/*.py", "**/tests/**/*.py"]
alwaysApply: false
---

# Testing Conventions

## Markers (declared in `pytest.ini` and `tests/nvidia/pytest.ini`)

| Marker | Use when |
|--------|----------|
| `asyncio` | Test function is `async def` |
| `integration` | Test touches a real external service (network, filesystem beyond tmp, subprocess) |
| `real_data` | Test depends on local datasets (e.g. `/lustre/.../train.parquet`) that may be absent on dev laptops |
| `slow` | Single-test wall clock > 10s |

### Default fast loop

`pytest -m "not integration and not slow and not real_data" tests/ -q`

This is what `/verify` and `/tdd` run. Keep new unit tests out of those markers so they land in the fast loop.

## Env vars required for runtime tests

If you're editing tests under `tests/runtime/**` or any test that exercises the sandbox:

```bash
export TEST_RUNTIME=singularity
export RUN_AS_OPENHANDS=False
export PYTHONPATH=.
```

## Fixtures and mocking

- Prefer fixtures over module-level globals; scope them tightly (`function` > `module` > `session`).
- Mock external HTTP via `responses` / `httpx.MockTransport` rather than patching `requests` directly.
- For reward-server-style remote dependencies: mock at the client boundary, not inside the production code path.
- For async: use `pytest-asyncio`; use `asyncio.wait_for(..., timeout=...)` to bound flaky awaits.

## Parameterization

- `@pytest.mark.parametrize` with named IDs: `ids=["happy", "empty-input", "error-on-bad-format"]`.
- Keep parametrized test inputs as literals at the top of the module or in a fixtures module — avoid computed fixtures when possible.

## Coverage expectations

- 80% minimum for new code landing in this repo.
- 100% for: reward-scorer logic, message-format/serialization code, token-handling paths, anything at the registry boundary.

## Anti-patterns

- Don't write tests that only assert the code runs ("smoke tests dressed as units") — assert observable behavior.
- Don't assert on log strings unless the log is part of the public contract.
- Don't mock what you own; refactor instead.
- Don't mark a slow test as `not slow` to get it into the fast loop — either make it fast or leave it marked.

See `skills/python-testing/SKILL.md` for deeper patterns.
