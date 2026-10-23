# Working on game-ui-magiclayer

This repository contains the standalone Python image decomposition algorithm.
Read README.md first and follow its automated setup sequence. Default to the
CPU SAM ViT-B demo; use scripts/bootstrap.py, scripts/doctor.py and scripts/run.py.
Do not assume a parent Server checkout, Java, Docker or a local API gateway exists.

- Never print, commit or upload .env, API keys, tokens, private configuration,
  input images, job output, virtual environments, model weights or third_party.
  Publish only empty configuration templates and model download metadata.
- Load local configuration through the launcher before importing algorithm modules.
- Use a new job ID for every run. Preserve previous output and failure evidence.
- If no API credentials are supplied, complete the local demo. Online mode needs
  user-provided endpoint, key and actual model IDs; do not invent them.
- README contains optional SAM2/SAM3 setup. Do not upgrade the main CPU environment
  to CUDA to enable SAM3; create its separate Python 3.12 environment.
- Run only verification appropriate to the new behavior. Existing algorithm tests
  are source references and are not automatically rerun during installation.
  Record new checks in doc/SETUP-VERIFICATION.md without overwriting failures.
- Before publishing, stage intended files explicitly, then run
  `python scripts/check_publish.py`. It scans the Git index and prints findings
  without disclosing matched secrets. Review the index before pushing.
