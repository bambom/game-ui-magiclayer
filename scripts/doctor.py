"""Check the interpreter, OCR bundle and SAM model without external API calls."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

from setup_support import ROOT, load_env, model_specs, valid_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=[*model_specs(), 'sam3'])
    args = ap.parse_args()
    load_env()
    errors = []
    for name in ['numpy', 'cv2', 'PIL', 'requests', 'onnxruntime', 'rapidocr_onnxruntime',
                 'torch', 'torchvision', 'segment_anything']:
        try:
            module = importlib.import_module(name)
            print(f'OK import {name}: {getattr(module, "__version__", "installed")}')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__} (run bootstrap with this Python version)')
    try:
        ocr = importlib.import_module('rapidocr_onnxruntime')
        models = Path(ocr.__file__).parent / 'models'
        for name in ['ch_PP-OCRv4_det_infer.onnx', 'ch_PP-OCRv4_rec_infer.onnx',
                     'ch_ppocr_mobile_v2.0_cls_infer.onnx']:
            if not (models / name).is_file():
                errors.append(f'OCR bundle missing {name}; reinstall rapidocr-onnxruntime==1.4.4')
    except Exception:
        # The first import loop already records the package/DLL failure.
        pass
    model = args.model or os.environ.get('MAGICLAYER_SAM_MODEL', 'vit_b')
    if model in model_specs():
        spec = model_specs()[model]
        if not valid_model(ROOT / 'algorithm/models' / spec['filename'], spec):
            errors.append(f'Model {model} missing or invalid; run bootstrap --model {model}')
        else:
            print(f'OK model {model}: size + SHA256')
        if model.startswith('sam2.'):
            try:
                importlib.import_module('sam2.sam2_image_predictor')
            except Exception as exc:
                errors.append(f'SAM2 dependency unavailable: {type(exc).__name__}')
    elif model == 'sam3':
        errors.append('SAM3 requires a separate CUDA environment. Follow README optional SAM3 setup; '
                      'this doctor validates the CPU bootstrap only.')
    else:
        errors.append('Unsupported MAGICLAYER_SAM_MODEL')
    if errors:
        for error in errors:
            print('ERROR ' + error, file=sys.stderr)
        return 1
    print('Environment ready (no online API request made).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
