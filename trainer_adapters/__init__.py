"""Trainer adapters (slot 5.6).

One sub-package per concrete adapter. Today: ``verl/``. Future: ``roll/``,
``slime/``.

Per principle 4.4 in ``rollout_fabric.md``, adapters compute their own
algorithm-specific fields locally; the wire schema (``schemas/training_sample.py``)
carries rewards, not advantages.
"""
