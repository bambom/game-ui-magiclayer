"""Audit Git's index before publishing; never print matched secret values."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'models', 'weights', 'checkpoints', 'jobs', 'uploads', 'third_party',
                '.local-notes', '__pycache__', '.cache', '.pytest_cache', 'tmp-batch'}
BLOCKED_SUFFIXES = {'.pt', '.pth', '.onnx', '.safetensors', '.ckpt', '.bin', '.h5', '.hdf5',
                    '.tflite', '.pb', '.pem', '.key', '.p12', '.pfx', '.pyc', '.part', '.tmp',
                    '.zip', '.gz', '.7z', '.log'}
PATTERNS = {
    'API credential': re.compile(r'(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|'
                                 r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
                                 r'AKIA[A-Z0-9]{16})'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'credential in URL': re.compile(r'https?://[^\s/"\']+:[^\s/"\']+@'),
    'literal credential': re.compile(
        r'(?i)["\']?(?:api[_-]?key|password|secret|access[_-]?token)["\']?\s*[:=]\s*'
        r'["\']([A-Za-z0-9_+/=.-]{16,})["\']'),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main() -> int:
    files = git('ls-files', '-s', '-z').split(b'\0')
    errors = []
    count, size = 0, 0
    for entry in files:
        if not entry:
            continue
        meta, raw_path = entry.split(b'\t', 1)
        path = raw_path.decode('utf-8')
        p = PurePosixPath(path)
        mode = meta.split()[0]
        count += 1
        if mode not in {b'100644', b'100755'}:
            errors.append(f'{path}: symlink/submodule/unmerged entry is not allowed')
            continue
        if (any(part in BLOCKED_DIRS or part.startswith('.venv') for part in p.parts)
                or (p.name.startswith('.env') and p.name != '.env.example')
                or p.suffix.lower() in BLOCKED_SUFFIXES):
            errors.append(f'{path}: excluded file type/directory')
        blob = git('show', ':' + path)
        size += len(blob)
        image = p.suffix.lower() in {'.png', '.gif', '.jpg', '.jpeg', '.webp'}
        limit = (25 if image else 5) * 1024 * 1024
        if len(blob) > limit:
            errors.append(f'{path}: larger than the source/document size limit')
        if b'\0' in blob:
            if p.suffix.lower() not in {'.png', '.gif', '.jpg', '.jpeg', '.webp'}:
                errors.append(f'{path}: unexpected binary file')
            continue
        try:
            content = blob.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'{path}: unexpected non-UTF8 file')
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    errors.append(f'{path}:{number}: {label} (value redacted)')
    for error in errors:
        print('ERROR ' + error, file=sys.stderr)
    print(f'Publish audit: {count} indexed files, {size / 1024 / 1024:.2f} MiB, {len(errors)} findings')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
