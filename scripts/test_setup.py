"""Incremental checks for standalone setup tools; no algorithm regression or API calls."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import bootstrap
import check_publish
import setup_support


class SetupChecks(unittest.TestCase):
    def test_download_and_reuse_without_second_request(self):
        payload = b'setup transport fixture, not a real model'
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as temp, patch.object(bootstrap, 'ROOT', Path(temp)):
                spec = {'filename': 'fixture.pt', 'size': len(payload),
                        'sha256': hashlib.sha256(payload).hexdigest(),
                        'url': f'http://127.0.0.1:{server.server_port}/fixture'}
                target = bootstrap.download_model(spec)
                self.assertEqual(target.read_bytes(), payload)
                self.assertFalse(target.with_suffix('.pt.part').exists())
                self.assertEqual(bootstrap.download_model(spec), target)
                self.assertEqual(len(calls), 1)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_corrupted_existing_model_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(bootstrap, 'ROOT', Path(temp)):
            path = Path(temp) / 'algorithm/models/fixture.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'existing-corrupt-file')
            with self.assertRaises(RuntimeError):
                bootstrap.download_model({'filename': path.name, 'size': 1, 'sha256': '0' * 64})
            self.assertEqual(path.read_bytes(), b'existing-corrupt-file')

    def test_bad_download_never_becomes_final_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(bootstrap, 'ROOT', Path(temp)):
            def fake_response(*_args, **_kwargs):
                result = io.BytesIO(b'wrong-checksum')
                result.status = 200
                return result
            spec = {'filename': 'fixture.pt', 'size': 14, 'sha256': '0' * 64,
                    'url': 'https://download.example/fixture'}
            with patch.object(bootstrap.urllib.request, 'urlopen', side_effect=fake_response), \
                    patch.object(bootstrap.time, 'sleep'), self.assertRaises(RuntimeError):
                bootstrap.download_model(spec)
            self.assertFalse((Path(temp) / 'algorithm/models/fixture.pt').exists())

    def test_local_env_does_not_override_process_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(setup_support, 'ROOT', Path(temp)):
            (Path(temp) / '.env').write_text('# comment\nML_SETUP_VALUE=file\nML_SETUP_NEW=value\n', encoding='utf-8')
            with patch.dict(os.environ, {'ML_SETUP_VALUE': 'process'}, clear=True):
                setup_support.load_env()
                self.assertEqual(os.environ['ML_SETUP_VALUE'], 'process')
                self.assertEqual(os.environ['ML_SETUP_NEW'], 'value')

    def test_publish_audit_blocks_forced_model_and_secret(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(check_publish, 'ROOT', Path(temp)):
            root = Path(temp)
            subprocess.run(['git', 'init', '-q', temp], check=True)
            (root / 'model.pt').write_bytes(b'weights')
            fake_secret = 'sk-' + 'A' * 32
            (root / 'settings.txt').write_text('API_KEY=' + fake_secret, encoding='utf-8')
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            captured = io.StringIO()
            with redirect_stderr(captured), redirect_stdout(captured):
                code = check_publish.main()
            self.assertEqual(code, 1)
            self.assertIn('model.pt', captured.getvalue())
            self.assertIn('settings.txt', captured.getvalue())
            self.assertNotIn(fake_secret, captured.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
