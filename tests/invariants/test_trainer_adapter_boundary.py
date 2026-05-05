"""Boundary condition: trainer_adapters/ modules must not import openhands, pandas,
parquet loaders, or vLLM. Trainer adapters own only the padding/collation seam
between LiveStore wire format and the trainer's native tensor format (BC-11, BC-15).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FORBIDDEN_PREFIXES = {'openhands', 'pandas', 'vllm', 'pyarrow.parquet'}
FORBIDDEN_EXACT = {'pandas', 'vllm', 'openhands'}


def _direct_imports_of(source_file: Path) -> set[str]:
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


def _collect_adapter_py_files() -> list[Path]:
    """Collect .py files under trainer_adapters/ (excluding __pycache__)."""
    adapters_dir = REPO_ROOT / 'trainer_adapters'
    if not adapters_dir.exists():
        return []
    return [p for p in adapters_dir.rglob('*.py') if '__pycache__' not in p.parts]


@pytest.mark.parametrize('py_file', _collect_adapter_py_files())
def test_trainer_adapter_does_not_import_forbidden_modules(py_file: Path) -> None:
    """trainer_adapters/**/*.py must not import openhands, pandas, vllm."""
    found = _direct_imports_of(py_file) & FORBIDDEN_EXACT
    assert not found, (
        f'{py_file.relative_to(REPO_ROOT)} imports {found}. '
        'Trainer adapters must own only the padding/collation seam — '
        'no openhands, no parquet loaders, no vLLM (BC-15).'
    )
