"""Boundary condition: EnvironmentProvider implementations must NOT import
live_store or policy_registry. Mirrors BC-13 (zero VERL/OpenHands imports in
RolloutManager) for the environment side.

An EnvironmentProvider that imports from live_store or policy_registry would
violate the service boundary — the environment is a pure data source, not an
orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FORBIDDEN_MODULES = {'live_store', 'policy_registry'}


def _direct_imports_of(source_file: Path) -> set[str]:
    """Return the set of top-level package names imported in a .py file."""
    try:
        text = source_file.read_text()
    except OSError:
        return set()
    imports: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('import '):
            top = stripped.split()[1].split('.')[0]
            imports.add(top)
        elif stripped.startswith('from '):
            parts = stripped.split()
            if len(parts) >= 2:
                top = parts[1].split('.')[0]
                imports.add(top)
    return imports


def _collect_env_provider_files() -> list[Path]:
    """Collect Python files under environment_providers/ (excluding __pycache__)."""
    ep_dir = REPO_ROOT / 'environment_providers'
    if not ep_dir.exists():
        return []
    return [p for p in ep_dir.rglob('*.py') if '__pycache__' not in p.parts]


@pytest.mark.parametrize('py_file', _collect_env_provider_files())
def test_environment_provider_does_not_import_fabric_internals(py_file: Path) -> None:
    """environment_providers/**/*.py must not import live_store or policy_registry."""
    found = _direct_imports_of(py_file) & FORBIDDEN_MODULES
    assert not found, (
        f'{py_file.relative_to(REPO_ROOT)} imports {found}. '
        'EnvironmentProvider implementations must not call LiveStore or PolicyRegistry.'
    )
