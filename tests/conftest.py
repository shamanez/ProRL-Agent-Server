"""Top-level pytest config for the rollout-fabric tests tree.

Exists so ``pytest tests/`` discovers ``tests/invariants/``,
``tests/contracts/``, and (later) ``tests/slots/`` without per-subdir
INI files.
"""

import sys
from pathlib import Path

# Make the repo root importable as a package root so ``schemas.*`` and
# ``live_store.*`` resolve when the tests run from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
