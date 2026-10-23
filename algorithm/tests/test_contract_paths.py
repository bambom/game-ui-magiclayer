"""Exercise the real output writer with small local pixels; no model execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from algorithm.common import ImageLayer, TextLayer, Plan
from algorithm.compose.compose import save_outputs
from algorithm.pipeline import Pipeline
from algorithm.storage import JobStore


class ContractPathTests(unittest.TestCase):
    def test_written_images_and_contract_use_portable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(out_dir=directory)
            original = np.zeros((2, 2, 3), dtype=np.uint8)
            rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
            layer = ImageLayer("picture", "Picture", "image", "element", 0, 0, 2, 2,
                               0, None, "", "local")
            text = TextLayer("text_001", "A", 0, 0, 2, 2, 1, {})
            save_outputs(directory, original, [(layer, rgba)], [text],
                         {text.region_id: rgba}, original)
            pipeline = SimpleNamespace(store=store, meta={}, warnings=[], timings={})
            plan = Plan("fixture", "scene", "fixture", [], [], [])
            with patch("algorithm.pipeline.runlog_enabled", return_value=False):
                Pipeline._write_contract(pipeline, original, [(None, rgba, layer)],
                                         [text], plan, {},
                                         SimpleNamespace(name="fixture", degraded=False), False)
            contract = json.loads(Path(store.layers_json).read_text(encoding="utf-8"))
            self.assertEqual([item["png"] for item in contract["layers"]],
                             ["layers/picture.png", "layers_text/text_001.png"])
            self.assertEqual(layer.png_rel, "layers/picture.png")
            for item in contract["layers"]:
                self.assertTrue((Path(directory) / item["png"]).is_file())


if __name__ == "__main__":
    unittest.main()
