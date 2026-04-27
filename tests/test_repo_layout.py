"""Smoke test: repo layout and PTQ4ViT config import."""

import os
import sys
import unittest


class TestRepoLayout(unittest.TestCase):
    def setUp(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.root not in sys.path:
            sys.path.insert(0, self.root)

    def test_framework_dirs(self):
        for name in ("configs", "quant_layers", "utils", "convit", "pvt"):
            path = os.path.join(self.root, name)
            self.assertTrue(os.path.isdir(path), f"missing dir: {path}")

    def test_entry_scripts(self):
        for rel in (
            "convit/apply_ptq.py",
            "convit/apply_ptq4vit.py",
            "convit/compare_bits_visual.py",
            "pvt/apply_ptq_pvt.py",
            "pvt/apply_ptq4vit_pvt.py",
        ):
            self.assertTrue(os.path.isfile(os.path.join(self.root, rel)), f"missing: {rel}")

    def test_configs_import(self):
        import configs.PTQ4ViT as cfg  # noqa: PLC0415

        self.assertTrue(hasattr(cfg, "get_module"))


if __name__ == "__main__":
    unittest.main()
