---
name: python-conventions
description: Python code conventions for this repo — Poetry, Ruff, Mypy, Pytest. Loaded only when editing .py files.
globs: ["**/*.py"]
alwaysApply: false
---

# Python Conventions

## Environment

- Python **3.12** (see `pyproject.toml`). Do not use syntax that requires 3.13+.
- Package manager: **Poetry**. Do not `pip install` into the venv; use `poetry add` / `poetry install --with <group>`.
- Groups declared in `pyproject.toml`: `dev`, `test`, `runtime`, `evaluation`. Install the groups you need, not all of them blindly.
- Two git deps must be installed out-of-band (they are not on PyPI in required form):
  `pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git`
  `pip install git+https://github.com/R2E-Gym/R2E-Gym.git`

## Linting and formatting — Ruff

- Config: `dev_config/python/ruff.toml`
- Run on touched files: `poetry run ruff check --config dev_config/python/ruff.toml --fix <path>`
- Format: `poetry run ruff format --config dev_config/python/ruff.toml <path>`
- Ruff also handles import sorting (don't reach for `isort`).

## Type checking — Mypy

- Config: `dev_config/python/mypy.ini`
- Full scan (slow, 15–30s): invoked by `make lint` via pre-commit (`always_run: true`, `pass_filenames: false`).
- Prefer completing the edit loop, then running `make lint` once — do **not** chain mypy after every edit.
- Add type hints to new code; leave existing untyped code as-is unless you're already editing it.

## Pre-commit

- Install once: `make build` (invokes `poetry run pre-commit install --config ./dev_config/python/.pre-commit-config.yaml`).
- Run on tracked files: `make lint`
- Run on scripts subtree: `make lint-scripts`
- **Never** commit with `--no-verify`. If a hook fails, fix the issue; don't skip the hook.

## Dependency pins

- `pyproject.toml` has load-bearing pins (comments explain the why). Examples include security CVEs and known bugs.
- Before widening any pin: read the comment on that line, grep the issue tracker, and ask before making the change.

## Pytest

- Config: `pytest.ini` (root) and `tests/nvidia/pytest.ini` (per-subtree).
- Loop per function is async; warnings are disabled.
- Available markers:
  - `asyncio` — async-def tests
  - `integration` — hit real services / real data
  - `real_data` — depend on local datasets that may not be present
  - `slow` — take >10s individually
- Fast iteration loop: `pytest -m "not integration and not slow and not real_data" tests/ -q`
- Coverage on a subtree: `pytest --cov=<module> --cov-report=term-missing tests/<module>/`
- Env vars sometimes required for runtime tests: `TEST_RUNTIME=singularity RUN_AS_OPENHANDS=False PYTHONPATH=.`

## Idiom pointers

- Prefer `dataclass` / `@dataclass(slots=True, frozen=True)` over ad-hoc classes with `__init__`.
- Prefer `asyncio.create_task` + structured cancellation over fire-and-forget loops.
- Use `pathlib.Path` over `os.path` for new code.
- Use `logging` (module-level `logger = logging.getLogger(__name__)`), not `print`, for production output.
- EAFP over LBYL when the happy path is the common case.
- When in doubt, read `skills/python-patterns/SKILL.md`.
