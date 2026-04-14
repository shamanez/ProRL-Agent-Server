---
name: repo-guardrails
description: Top-line invariants that apply to every file in this repo. Kept deliberately short; deeper conventions live in glob-gated rules and skills.
globs: ["**"]
alwaysApply: true
---

# Repo Guardrails

- **Read `CLAUDE.md` first.** It names the top-level abstractions, the invariants, and the out-of-scope list.
- **Never commit with `--no-verify`.** If a hook is failing, fix the underlying issue.
- **Before first edit in an unfamiliar module**, apply the `repo-architecture` skill.
- **Do not modify** linter/formatter/type-checker configs (`dev_config/python/**`) without explicit user approval.
- **Do not widen pinned dependencies** in `pyproject.toml` without reading the pin comment and asking.
- **Singularity/Apptainer is the default sandbox runtime** where applicable; Docker exists as a fallback path.
- **`make lint`** runs pre-commit (ruff + mypy + pyproject-fmt). Run it before declaring a change ready.
- Fast test loop: `pytest -m "not integration and not slow and not real_data" tests/ -q`.

When a task is non-trivial, prefer `/plan` first. When unsure which skill applies, check `.claude/skills/` by name — or read `repo-architecture` for orientation.
