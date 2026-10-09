"""Integration checks for portable commands, matched budgets and vote aggregation."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run

spec = importlib.util.spec_from_file_location("score_varc", ROOT / "scripts" / "score_varc.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class ReleaseTests(unittest.TestCase):
    def test_varc_commands_parse_with_native_parser(self):
        spec = importlib.util.spec_from_file_location("native_varc_args", ROOT / "vendor" / "varc" / "utils" / "args.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        original = sys.argv
        try:
            for stage, configs in [("pretrain", run.BUDGETS["pretraining"]), ("ttt", run.BUDGETS["ttt"])]:
                for config in configs:
                    a = run.parse_args(["varc", stage, "--config", config])
                    cmd, _ = run.build_command(a, "00576224" if stage == "ttt" else None)
                    script = "offline_train_ARC.py" if stage == "pretrain" else "test_time_train_ARC.py"
                    sys.argv = cmd[cmd.index(script):]
                    parsed = native.parse_args()
                    self.assertEqual(parsed.epochs, run.BUDGETS["pretraining" if stage == "pretrain" else "ttt"][config]["varc_epochs"])
                    self.assertFalse(hasattr(parsed, "include_rearc"))
                    self.assertEqual(parsed.architecture, "vit")
        finally:
            sys.argv = original

    def test_all_commands_parse_in_native_harnesses(self):
        # Native TRM wrappers are stdlib-only: dry runs exercise actual command building.
        with tempfile.TemporaryDirectory() as tmp:
            for stage, configs in [("pretrain", run.BUDGETS["pretraining"]), ("ttt", run.BUDGETS["ttt"])]:
                for config in configs:
                    args = ["trm", stage, "--config", config, "--output", str(Path(tmp) / config / stage)]
                    if stage == "ttt":
                        args += ["--task-id", "00576224"]
                    a = run.parse_args(args)
                    cmd, cwd = run.build_command(a, a.task_id)
                    result = subprocess.run(cmd + ["--dry-run"], cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("pretrain.py", result.stdout)

    def test_pretraining_epoch_conversion(self):
        for c in run.BUDGETS["pretraining"].values():
            self.assertEqual(c["varc_epochs"], math.ceil(c["target_presentations"] / (1302 * c["variants"])))
            self.assertEqual(c["trm_epochs"] % c["trm_eval_interval"], 0)

    def test_varc_scratch_and_pretrained_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.touch()
            a = run.parse_args(["varc", "ttt", "--config", "v51_low", "--task-id", "00576224"])
            cmd, _ = run.build_command(a, a.task_id)
            self.assertIn("eval_color_permute_ttt_9/00576224", cmd)
            a.checkpoint = checkpoint
            cmd, _ = run.build_command(a, a.task_id)
            self.assertIn("--resume-skip-task-token", cmd)
            self.assertIn("--ttt-on-the-fly-augmentations", cmd)

    def test_vote_ties_and_task_weighting(self):
        task = {"test": [{"output": [[1]]}, {"output": [[2]]}]}
        score = scorer.score_task(task, {"0": [[[0]], [[1]]], "1": [[[2]], [[2]], [[0]]]})
        self.assertEqual(score, {"pass_at_1": 0.5, "pass_at_2": 1., "oracle": 1.})
        with self.assertRaises(ValueError):
            scorer.score_task(task, {"0": [[[1]]]})

    def test_only_paper_architectures_and_dataset_builders_are_bundled(self):
        self.assertEqual({p.name for p in (ROOT / "vendor/varc/src").glob("*.py")}, {"ARC_ViT.py", "ARC_loader.py"})
        self.assertEqual({p.name for p in (ROOT / "vendor/trm/models/recursive_reasoning").glob("*.py")}, {"trm.py"})
        self.assertEqual({p.name for p in (ROOT / "vendor/trm/config/arch").glob("*.yaml")}, {"trm.yaml"})
        self.assertEqual({p.name for p in (ROOT / "vendor/trm/dataset").glob("*.py")}, {"build_arc_dataset.py", "common.py"})
        forbidden = ("ARCPrompt", "ARCVisual", "ARCFlow", "ARCCanvasFlow", "ARCLoop", "ARCDirectOneHot", "vit_path", "include_rearc", "get_augmenters")
        for p in (ROOT / "vendor/varc").rglob("*.py"):
            for term in forbidden:
                self.assertNotIn(term, p.read_text(), str(p))

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "missing"
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "varc", "ttt", "--config", "v1001_high", "--task-id", "00576224", "--output", str(dest), "--dry-run"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
