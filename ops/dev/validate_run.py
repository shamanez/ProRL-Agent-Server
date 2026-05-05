#!/usr/bin/env python3
"""Stage 2 decoupled-run post-hoc gate.

Parses BOTH the trainer logfile (``--trainer-log``, default
``/tmp/s2-decoupled.log``) AND the external supervisor child logs
(``--child-logs``, default ``/tmp/vllm-child-{8100,8101,8102,8103}.log``)
so the gates actually see both sides of the decoupling.

Trainer-log gates:
  (a) Literal ``EXTERNAL BYPASS ACTIVE`` WARNING present.
  (b) Zero Ray vLLM actor mentions (``async_llm_server_<n>``) — the trainer
      must not have spawned its own colocated pool.
  (d) ``global_step`` reached ``--min-global-step`` (default 20).
  (e) Metric sanity:
      e1. ``grad_norm`` finite and within ``(0, 1e6)``,
      e2. ``kl``       finite,
      e3. ``advantage`` has non-zero sample variance (not just any non-zero
          single value — variance is computed across all samples).

External-pool gates (read from the supervisor child logs):
  (c) Exactly ``--expect-weight-publishes`` ``POST /reload_weights`` lines
      across all child logs (default 0 = Stage 2 stale-weight invariant).
      This is the AUTHORITATIVE publish count because it's observed on the
      receiver side; a publish attempted but never delivered would still
      show up in the trainer log but not here, and vice versa.
  (f) Every child log shows at least one ``POST /generate`` hit — proves
      each external endpoint actually served rollout traffic.

Exit 0 on all-green. Non-zero with a summary table otherwise.

Usage::

    python scripts/validate_run.py
    python scripts/validate_run.py --trainer-log /tmp/s2-decoupled.log \
        --child-logs /tmp/vllm-child-8100.log /tmp/vllm-child-8101.log \
                     /tmp/vllm-child-8102.log /tmp/vllm-child-8103.log
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path


def _find_marker(text: str) -> str | None:
    if 'EXTERNAL BYPASS ACTIVE' not in text:
        return 'missing literal "EXTERNAL BYPASS ACTIVE" marker'
    return None


def _find_ray_vllm_actors(text: str) -> str | None:
    spawns = re.findall(r'async_llm_server_\d+', text)
    if spawns:
        unique = sorted(set(spawns))
        return f'found {len(spawns)} Ray vLLM actor mentions: {unique[:4]}'
    return None


def _count_weight_publishes_in_child_logs(
    child_logs: list[Path], expected: int
) -> str | None:
    """Authoritative publish count: ``POST /reload_weights`` across child logs.

    Uvicorn access-logs every POST the supervisor receives, including
    /reload_weights (which responds 501 in Stage 1/2). Counting on the
    receiver side is the only way to prove the stale-weight invariant —
    a trainer-side "I'm about to publish" log that never actually hits the
    wire would be a false positive.
    """
    missing = [p for p in child_logs if not p.exists()]
    if missing:
        return f'child logs missing: {[str(p) for p in missing]}'
    total = 0
    per_log: dict[str, int] = {}
    for p in child_logs:
        n = p.read_text(errors='replace').count('POST /reload_weights')
        per_log[p.name] = n
        total += n
    if total != expected:
        return (
            f'expected exactly {expected} /reload_weights across child logs, '
            f'got {total} ({per_log})'
        )
    return None


def _check_generate_traffic(child_logs: list[Path]) -> str | None:
    """Every child must have served at least one ``POST /generate``."""
    missing = [p for p in child_logs if not p.exists()]
    if missing:
        return f'child logs missing: {[str(p) for p in missing]}'
    per_log = {
        p.name: p.read_text(errors='replace').count('POST /generate')
        for p in child_logs
    }
    dead = [name for name, n in per_log.items() if n == 0]
    if dead:
        return (
            f'child logs with zero POST /generate: {dead} (full counts: {per_log}) '
            f'— at least one endpoint never served rollouts'
        )
    return None


def _max_global_step(text: str) -> int:
    steps = re.findall(r'global[_-]step[\s=:]+(\d+)', text, re.IGNORECASE)
    return max((int(s) for s in steps), default=0)


def _check_global_step(text: str, minimum: int) -> str | None:
    seen = _max_global_step(text)
    if seen < minimum:
        return f'max global_step={seen} < required {minimum}'
    return None


# Matches ``name: value`` / ``name=value`` with either a numeric value (incl.
# scientific notation) or a non-finite literal (``nan`` / ``inf`` / ``-inf``,
# any case). Without the non-finite alternative, a NaN grad_norm would be
# silently dropped and `_check_metric_finite` would report PASS — masking
# the exact divergence this gate is meant to catch.
_METRIC_RE = re.compile(
    r'([A-Za-z_][\w/]*)\s*[:=]\s*'
    r'([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))',
    re.IGNORECASE,
)


def _collect_metric(text: str, key: str) -> list[float]:
    vals: list[float] = []
    for name, raw in _METRIC_RE.findall(text):
        if name == key or name.endswith('/' + key):
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    return vals


def _check_grad_norm(text: str) -> str | None:
    """grad_norm must exist, be finite every time, and stay in (0, 1e6)."""
    vals = _collect_metric(text, 'grad_norm')
    if not vals:
        return 'no grad_norm values found in logfile'
    bad_nan = [v for v in vals if not math.isfinite(v)]
    if bad_nan:
        return (
            f'grad_norm non-finite: first bad value {bad_nan[0]} (total {len(bad_nan)})'
        )
    bad_bounds = [v for v in vals if not (0.0 < v < 1e6)]
    if bad_bounds:
        return (
            f'grad_norm out of bounds (0, 1e6): first offender {bad_bounds[0]} '
            f'(total {len(bad_bounds)} of {len(vals)})'
        )
    return None


def _check_metric_finite(text: str, key: str) -> str | None:
    vals = _collect_metric(text, key)
    if not vals:
        return f'no {key} values found in logfile (did the trainer log it?)'
    bad = [v for v in vals if not math.isfinite(v)]
    if bad:
        return f'{key} non-finite: first bad value {bad[0]} (total {len(bad)})'
    return None


def _check_advantage_variance(text: str) -> str | None:
    """Advantages must have real sample variance, not just one non-zero.

    A GRPO run where every advantage is identical (all zero, all 1) produces
    zero variance and therefore zero policy gradient — the update signal is
    dead. We require statistics.pvariance > 1e-8 across at least 2 samples.
    """
    vals = _collect_metric(text, 'advantage') + _collect_metric(text, 'advantages')
    if not vals:
        return 'no advantage/advantages values found in logfile'
    if len(vals) < 2:
        return f'only {len(vals)} advantage sample(s); need at least 2 for variance'
    variance = statistics.pvariance(vals)
    if variance <= 1e-8:
        return (
            f'advantage variance {variance:.3e} too small over {len(vals)} '
            f'samples (dead GRPO signal)'
        )
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--trainer-log',
        type=Path,
        default=Path('/tmp/s2-decoupled.log'),
        help='Trainer logfile (default: %(default)s).',
    )
    parser.add_argument(
        '--child-logs',
        type=Path,
        nargs='+',
        default=[Path(f'/tmp/vllm-child-{p}.log') for p in (8100, 8101, 8102, 8103)],
        help='External vLLM child logs (default: /tmp/vllm-child-810{0..3}.log).',
    )
    parser.add_argument(
        '--expect-weight-publishes',
        type=int,
        default=0,
        help='Exact number of POST /reload_weights across child logs (default: 0).',
    )
    parser.add_argument(
        '--min-global-step',
        type=int,
        default=20,
        help='Minimum global_step reached (default: 20).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.trainer_log.exists():
        print(f'ERROR: trainer log not found: {args.trainer_log}', file=sys.stderr)
        return 2
    text = args.trainer_log.read_text(errors='replace')

    gates: list[tuple[str, str | None]] = [
        ('a. EXTERNAL BYPASS ACTIVE marker', _find_marker(text)),
        ('b. zero Ray vLLM actor spawns', _find_ray_vllm_actors(text)),
        (
            f'c. exactly {args.expect_weight_publishes} /reload_weights in child logs',
            _count_weight_publishes_in_child_logs(
                args.child_logs, args.expect_weight_publishes
            ),
        ),
        (
            f'd. global_step >= {args.min_global_step}',
            _check_global_step(text, args.min_global_step),
        ),
        ('e1. grad_norm finite and in (0, 1e6)', _check_grad_norm(text)),
        ('e2. kl finite', _check_metric_finite(text, 'kl')),
        ('e3. advantage variance > 1e-8', _check_advantage_variance(text)),
        (
            'f. every child log served POST /generate',
            _check_generate_traffic(args.child_logs),
        ),
    ]

    print()
    print(f'Stage 2 validate_run.py — trainer={args.trainer_log}')
    print(f'  child logs: {[str(p) for p in args.child_logs]}')
    print('=' * 72)
    fails = 0
    for label, err in gates:
        status = ' PASS ' if err is None else ' FAIL '
        line = f'[{status}] {label}'
        if err:
            line += f'  — {err}'
            fails += 1
        print(line)
    print('=' * 72)
    print(f'{len(gates) - fails}/{len(gates)} gates passed')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
