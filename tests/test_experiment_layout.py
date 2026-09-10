import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.synthetic.run import main as synthetic
from experiments.real.run import main as real
from experiments.real.extract import main as extract


class ExperimentLayoutTests(unittest.TestCase):
    def test_synthetic_config_override_and_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"synthetic.json"
            synthetic(["--config", "experiments/synthetic/configs/full.json", "--epochs", "1", "--out", str(out)])
            report = json.loads(out.read_text())
            self.assertEqual(report["experiment_family"], "synthetic")
            self.assertEqual(report["epochs"], 1)
            self.assertEqual(report["readout_rank"], 3)
            self.assertEqual(report["unconstrained_dim"], 0)
            self.assertTrue(out.with_suffix(".pt").exists())
            events = EventAccumulator(report["tensorboard_log_dir"]).Reload()
            self.assertIn("synthetic-", Path(report["tensorboard_log_dir"]).name)
            for tag in ("train/loss", "train/task_accuracy", "eval/loss", "eval/task_accuracy",
                        "audit/equivalence/max_logit_error", "audit/consequence/per_concept/0/accuracy_drop"):
                self.assertEqual(len(events.Scalars(tag)), 1)
                self.assertEqual(events.Scalars(tag)[0].step, 1)
            self.assertAlmostEqual(events.Scalars("train/loss")[0].value,
                                   events.Scalars("train/concept_loss")[0].value + events.Scalars("train/task_loss")[0].value,
                                   places=6)

    def test_real_has_no_synthetic_fallback(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            real([])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            synthetic(["--config", "experiments/real/configs/cub_dinov2.json"])

    def test_real_separate_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(4)
            for split, n in (("train", 20), ("test", 8)):
                p = root/split
                p.mkdir()
                z = rng.normal(size=(n, 5)).astype("float32")
                np.save(p/"embeddings.npy", z)
                np.save(p/"concepts.npy", (z[:, :2] > 0).astype("float32"))
                np.save(p/"labels.npy", (z[:, 2] > 0).astype("int64"))
            real(["--features-dir", str(root/"train"), "--eval-features-dir", str(root/"test"),
                           "--epochs", "1", "--out", str(root/"real.json"), "--log-dir", str(root/"logs")])
            report = json.loads((root/"real.json").read_text())
            self.assertEqual(report["experiment_family"], "real")
            self.assertEqual(report["train_samples"], 20)
            self.assertEqual(report["eval_samples"], 8)
            self.assertEqual(Path(report["tensorboard_log_dir"]).parent, root/"logs")
            self.assertIn("real-", Path(report["tensorboard_log_dir"]).name)
            events = EventAccumulator(report["tensorboard_log_dir"]).Reload()
            self.assertTrue(events.Scalars("eval/task_accuracy"))

    def test_extraction_defaults_to_real_feature_folder(self):
        with patch("experiments.real.extract.extract") as call:
            extract(["--dataset", "cifar10"])
            self.assertEqual(call.call_args.args[0][-2:], ["--out-dir", "features/real/default"])


if __name__ == "__main__":
    unittest.main()
