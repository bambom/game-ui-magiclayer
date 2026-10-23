"""Standard-library helpers shared by the standalone setup and launch tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    path = ROOT / '.env'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


def venv_python(directory: Path) -> Path:
    return directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def model_specs() -> dict:
    return json.loads((ROOT / 'config/models.json').read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_model(path: Path, spec: dict) -> bool:
    return (path.is_file() and path.stat().st_size == spec['size']
            and sha256(path) == spec['sha256'])
