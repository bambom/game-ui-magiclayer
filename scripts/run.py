"""Start a local demo or process an image; load .env before importing the algorithm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from setup_support import ROOT, load_env, model_specs


def create_demo(directory: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (384, 384), (225, 232, 245))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 64, 344, 320), radius=18, fill=(74, 104, 160))
    draw.ellipse((136, 120, 248, 232), fill=(245, 190, 60))
    draw.rounded_rectangle((128, 264, 256, 300), radius=10, fill=(70, 190, 130))
    image_path, plan_path = directory / 'input.png', directory / 'plan.json'
    image.save(image_path)
    queries = []
    for name, kind, box, parent, z in [
        ('panel', 'panel', [40, 64, 344, 320], None, 1),
        ('badge', 'badge', [136, 120, 248, 232], 'panel', 2),
        ('button', 'button', [128, 264, 256, 300], 'panel', 3),
    ]:
        x0, y0, x1, y1 = box
        queries.append({
            'id': name, 'name': name, 'kind': kind,
            'role': 'container' if parent is None else 'foreground',
            'element_repair_mode': 'surface' if parent is None else 'none',
            'parent_query_id': parent, 'z_order': z,
            'geometry_hints': [{'bbox_norm': [v / 384 for v in box],
                                'positive_points_norm': [[(x0 + x1) / 768, (y0 + y1) / 768]],
                                'negative_points_norm': []}],
        })
    plan_path.write_text(json.dumps({
        'scene_summary': 'Synthetic panel with badge and button',
        'background_repair': {'mode': 'surface', 'reason': 'Solid background'},
        'text_corrections': [], 'raster_text_ids': [], 'queries': queries,
    }, indent=2), encoding='utf-8')
    return image_path, plan_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Generated image + fixed plan + local repair, no API')
    source.add_argument('--image', type=Path)
    ap.add_argument('--plan', type=Path, help='Local planning JSON; otherwise use online planning')
    ap.add_argument('--local', action='store_true', help='Use local repair (requires --plan for a real image)')
    ap.add_argument('--model', choices=[*model_specs(), 'sam3'], help='Override .env SAM backend')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--jobs-root', type=Path, default=ROOT / 'jobs')
    ap.add_argument('--job-id', help='New job directory name (default generates a unique ID)')
    args = ap.parse_args()
    load_env()  # config.py reads numeric overrides at import time.
    if args.model:
        os.environ['MAGICLAYER_SAM_MODEL'] = args.model
    job_id = args.job_id or ('demo-' if args.demo else 'run-') + uuid.uuid4().hex[:12]
    safe = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-'
    if not job_id or len(job_id) > 120 or any(c not in safe for c in job_id) or job_id in {'.', '..'}:
        ap.error('--job-id must be a safe directory name (letters, digits, dot, dash, underscore; max 120)')
    jobs_root = args.jobs_root.resolve()
    output = jobs_root / job_id
    if output.exists():
        ap.error('Job already exists. Choose a new --job-id; existing results were not overwritten.')
    local = args.local or args.demo
    image, plan = args.image, args.plan
    if args.demo:
        image, plan = create_demo(jobs_root / ('_demo-input-' + uuid.uuid4().hex[:12]))
    elif not image.is_file():
        ap.error('--image does not exist')
    if plan is not None and not plan.is_file():
        ap.error('--plan does not exist')
    if local and plan is None:
        ap.error('--local requires --plan; use --demo for an automatic example')
    if plan is None:
        key = (os.environ.get('MAGICLAYER_PLANNER_KEY') or os.environ.get('GEMINI_API_KEY')
               or os.environ.get('OPENAI_API_KEY') or os.environ.get('MAGICLAYER_GEMINI_KEY'))
        if not key or not os.environ.get('MAGICLAYER_PLANNER_MODEL', '').strip():
            ap.error('Online planning needs an API key and MAGICLAYER_PLANNER_MODEL. '
                     'Set them locally per .env.example, or use --demo.')
    command = [sys.executable, '-m', 'algorithm.pipeline', '--image', str(image.resolve()),
               '--job-id', job_id, '--jobs-root', str(jobs_root), '--device', args.device,
               '--planner', 'mock' if plan else 'gemini', '--provider', 'local' if local else 'gemini']
    if plan:
        command += ['--plan', str(plan.resolve())]
    print(f'Starting job {job_id}; mode={"local" if local else "online"}', flush=True)
    result = subprocess.run(command, cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    if result.returncode:
        return result.returncode
    status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    contract = json.loads((output / 'layers.json').read_text(encoding='utf-8'))
    if status.get('state') != 'done' or not contract.get('layers'):
        print('Job did not produce a completed layer contract.', file=sys.stderr)
        return 1
    print(f'Completed: {len(contract["layers"])} layers\nPreview: {output / "preview.png"}\nContract: {output / "layers.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
