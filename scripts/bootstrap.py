"""Create a private Python environment and fetch verified official SAM weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import venv

from setup_support import ROOT, load_env, model_specs, valid_model, venv_python

TORCH = ('torch==2.6.0', 'torchvision==0.21.0')
TORCH_INDEX = 'https://download.pytorch.org/whl/cpu'
SAM2_REF = '2b90b9f5ceec907a1c18123530e92e794ad901a4'


def run(command: list[str], **kwargs) -> None:
    # Do not print environment variables, which may contain credentials.
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def download_model(spec: dict) -> Path:
    target = ROOT / 'algorithm/models' / spec['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_model(target, spec):
        print(f'Model already verified: {target.name}', flush=True)
        return target
    if target.exists():
        raise RuntimeError(f'{target.name}: existing file failed size/SHA256 check. '
                           'Move it aside explicitly before retrying; it was not overwritten.')
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(1, 4):
        try:
            print(f'Downloading {target.name} ({spec["size"] / 1e6:.0f} MB), attempt {attempt}/3',
                  flush=True)
            request = urllib.request.Request(spec['url'], headers={'User-Agent': 'magiclayer-setup/1'})
            with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as output:
                if response.status != 200:
                    raise RuntimeError('Model server did not return a full file (HTTP 200)')
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if not valid_model(part, spec):
                raise RuntimeError('Downloaded model failed size/SHA256 verification')
            part.replace(target)
            return target
        except Exception as exc:
            # URL/proxy errors can contain credentials; log only the exception type.
            print(f'Download attempt failed: {type(exc).__name__}', flush=True)
            if attempt == 3:
                raise RuntimeError('Model download failed. Check network/proxy; retry bootstrap. '
                                   'A .part file is never loaded as a model.') from None
            time.sleep(2)
    raise AssertionError('unreachable')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--venv', type=Path, default=ROOT / '.venv', help='Dedicated environment directory')
    ap.add_argument('--model', choices=list(model_specs()), default='vit_b')
    ap.add_argument('--download-only', action='store_true', help='Download/verify model only; skip pip')
    args = ap.parse_args()
    load_env()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12) or platform.architecture()[0] != '64bit':
        raise SystemExit('Use 64-bit Python 3.10, 3.11 or 3.12 for the CPU environment.')
    if args.download_only:
        download_model(model_specs()[args.model])
        return
    directory = args.venv.resolve()
    if not directory.exists():
        print(f'Creating environment: {directory}', flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    python = venv_python(directory)
    if not python.is_file() or not (directory / 'pyvenv.cfg').is_file():
        raise SystemExit('The environment path exists but is not a venv. Use an empty/dedicated path.')
    # Repeated bootstrap calls keep a successfully installed environment.
    fingerprint = hashlib.sha256((ROOT / 'requirements.txt').read_bytes()
                                 + Path(__file__).read_bytes()
                                 + args.model.encode() + sys.version.encode()).hexdigest()
    marker = directory / '.setup-state.json'
    installed = json.loads(marker.read_text()) if marker.is_file() else {}
    if installed.get('fingerprint') != fingerprint:
        run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
        torch_command = [str(python), '-m', 'pip', 'install', *TORCH]
        if sys.platform != 'darwin':
            torch_command += ['--index-url', TORCH_INDEX]
        run(torch_command)
        run([str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'])
        if args.model.startswith('sam2.'):
            env = dict(os.environ, SAM2_BUILD_CUDA='0')
            run([str(python), '-m', 'pip', 'install', '--no-build-isolation',
                 f'git+https://github.com/facebookresearch/sam2.git@{SAM2_REF}'], env=env)
        run([str(python), '-m', 'pip', 'check'])
        marker.write_text(json.dumps({'fingerprint': fingerprint, 'model': args.model}), encoding='utf-8')
    download_model(model_specs()[args.model])
    if not (ROOT / '.env').exists():
        shutil.copyfile(ROOT / '.env.example', ROOT / '.env')
    env = dict(os.environ, MAGICLAYER_SAM_MODEL=args.model)
    run([str(python), 'scripts/doctor.py', '--model', args.model], env=env)
    print('Ready. Run: ' + str(python) + f' scripts/run.py --demo --model {args.model}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
