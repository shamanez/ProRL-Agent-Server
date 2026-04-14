---
name: repo-architecture
description: Orient yourself in an unfamiliar module of a large multi-subsystem repo before editing. Read CLAUDE.md first, locate the entry point, identify the top-level abstractions, map module boundaries, and list edit groups (files that must be read together). Generic — not tied to any specific codebase.
when_to_use:
  - Opening a file or module you have not touched before
  - About to propose a non-trivial change in a subsystem
  - Triangulating between two related files that share a name or concept
  - A user asks "how does X work here?" and X is bigger than a single file
---

# repo-architecture — How to orient in a large codebase

This skill is about **not guessing**. Large research and production codebases accumulate subsystem-specific invariants (async pipelines, registries, dual dispatch paths, pinned external patches, framework extensions). A single-file read is rarely enough context to safely edit such modules.

## When to apply

Apply this skill **before** the first Edit/Write in any module you have not previously read. If you're in a module you have already mapped, skip it.

## The 5-step orientation

### 1. Read `CLAUDE.md` if present

Every repo that uses Claude Code should have a `CLAUDE.md` at the root. It will almost always:
- Name the top-level abstractions
- Point at the entry points (server launcher, main module, CLI dispatcher)
- Call out invariants ("never re-encode token IDs", "this pin is load-bearing")
- List conventions (package manager, linter, test runner, markers)

If there is no `CLAUDE.md`, escalate to reading `README.md` and the most recent commits on the main branch. Then propose writing a `CLAUDE.md` while context is fresh.

### 2. Locate the entry point of the relevant subsystem

Every non-trivial subsystem has one file that:
- Imports most of the subsystem's other files
- Exposes the public API (HTTP routes, `__init__.py` re-exports, CLI subcommands)
- Is the first target of new feature work

Find it by:
- `grep -l "app = FastAPI" server/` — framework entry points
- `grep -l "register_" core/` — registry bootstrap
- `grep -l "if __name__" scripts/` — CLI entry points
- Module `__init__.py` re-export lists

### 3. Identify the top-level abstractions

In 30 seconds, answer for yourself:
- What is the **central dataclass or protocol** that flows through the subsystem? (`JobDetails`, `Request`, `Context`, `Event`)
- What is the **dispatch mechanism**? (registry, router table, strategy pattern, chain of responsibility)
- What is the **lifecycle**? (create → queue → process → evict; init → run → eval; load → transform → save)
- What are the **worker/actor abstractions**? (thread pool, async workers, Ray actors, subprocess)

Write these down — you'll rely on them for every subsequent edit.

### 4. Map module boundaries

For any change, know:
- Which files in this subsystem does my change touch?
- Which files outside the subsystem import from the files I'm changing?
- Is there an upstream fork or vendored code in the tree? (Often a `third_party/`, `vendor/`, or `<framework>_custom/` directory.) Changes there often need to match an upstream contract.
- Are there parallel hierarchies? (E.g., `nvidia/`, `intel/`, `cpu/`; or `v1/`, `v2/`; or `default/`, `reasoning/`.) Know which path your change belongs in, and whether the other paths need a parallel change.

### 5. List edit groups

An **edit group** is a set of files that must be read together to make a safe change. Common patterns:

- **Dispatch triad**: `registry.py` ↔ `server/router.py` ↔ any concrete handler. Changing the handler signature breaks the registry.
- **Paired clients**: `client_a.py` + `client_b.py` (both implementations of the same protocol). Changing one without the other causes silent divergence.
- **Schema + validator**: `schema.py` + `validate.py`. Any field rename must propagate.
- **Config + loader + test fixture**: `config.toml` + `config_loader.py` + `tests/fixtures/test_config.toml`.
- **Same-name collisions**: two files with the same basename in different directories that serve different roles (e.g. a rollout-manager `server.py` vs. a FastAPI `server.py`). Name a specific suspected collision before editing so you don't confuse them.

Add any edit groups you discover to `CLAUDE.md` under an "Edit groups" section so the next person (or the next Claude session) doesn't have to rediscover them.

## Anti-patterns

- **Editing before reading.** A single-file read gives you the syntax but not the invariants. The invariants live in the neighbors.
- **Treating parallel paths as copy-paste.** If `nvidia/` and `cpu/` exist as parallel module trees, they are often NOT interchangeable — one may have state the other doesn't, one may have threading constraints the other doesn't.
- **Ignoring pinned third-party commits.** If you see `git checkout <sha>` in README or Dockerfile, any patch-package directory is implicitly pinned to that SHA. Bumping the pin is a separate, deliberate change.
- **Skipping `CLAUDE.md` "out of scope" sections.** These are often the most valuable signal about what NOT to touch.

## Deliverables after orientation

After applying this skill, you should be able to answer, for the subsystem at hand:

1. Entry point — one file.
2. Top-level abstraction — one dataclass/protocol + lifecycle sentence.
3. Dispatch — where does a new {request, task, event} get routed?
4. Edit groups — 2–5 file pairs that must move together.
5. Don't-touch list — config, pinned deps, vendored third-party code.

If any answer is still fuzzy after 10 minutes, escalate — ask the user or `/plan` it. Fuzzy answers produce wrong edits.

## Related

- `skills/strategic-compact` — compact session before deep exploration so full-tree reads don't crash your context
- Command `/plan` — if orientation surfaces that the requested change is bigger than first thought, re-plan
- Command `/context-budget` — audit what's in context before a big read sweep
