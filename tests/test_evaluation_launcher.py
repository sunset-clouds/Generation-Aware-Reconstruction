"""Exercise the shell launcher without GPU jobs or FID dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EvaluationLauncherTests(unittest.TestCase):
    def run_launcher(self, overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = (
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['GAR_TEST_COMMANDS'], 'a') as output:\n"
                "    output.write(json.dumps(sys.argv) + '\\n')\n"
            )
            for name in ("torchrun", "fid-python"):
                executable = root / name
                executable.write_text(recorder)
                executable.chmod(0o755)
            log = root / "commands.jsonl"
            env = {key: value for key, value in os.environ.items()
                   if key not in ("CFG_OMEGA", "CFG_T_MIN", "CFG_T_MAX")}
            env.update(PATH=str(root) + os.pathsep + env["PATH"],
                       PYTHON=str(root / "fid-python"), GAR_TEST_COMMANDS=str(log))
            env.update(overrides)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/evaluate_decoder.sh"), "iMF-XL-2",
                 "imf.pth", "decoder.pth.tar", "imagenet/val", "out with spaces", "ref.npz"],
                env=env, capture_output=True, text=True,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_default_uses_model_registry_without_cfg_overrides(self):
        result, calls = self.run_launcher({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--cfg_omega", calls[0])
        self.assertNotIn("--cfg_t_min", calls[0])
        self.assertNotIn("--cfg_t_max", calls[0])
        self.assertEqual(calls[0][calls[0].index("--model_type") + 1], "iMF-XL-2")
        self.assertIn("out with spaces", calls[0])
        self.assertIn("out with spaces/fid_results_openai.json", calls[1])

    def test_explicit_cfg_overrides_are_forwarded(self):
        result, calls = self.run_launcher({"CFG_OMEGA": "8", "CFG_T_MIN": "0.42", "CFG_T_MAX": "0.62"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag, value in (("--cfg_omega", "8"), ("--cfg_t_min", "0.42"), ("--cfg_t_max", "0.62")):
            self.assertEqual(calls[0][calls[0].index(flag) + 1], value)

    def test_partial_interval_fails_before_launching_jobs(self):
        for overrides in ({"CFG_T_MIN": "0.42"}, {"CFG_T_MAX": "0.62"}):
            with self.subTest(overrides=overrides):
                result, calls = self.run_launcher(overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("set CFG_T_MIN and CFG_T_MAX together", result.stderr)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
