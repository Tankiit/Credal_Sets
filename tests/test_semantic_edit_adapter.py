import tempfile
import unittest
from pathlib import Path
import torch
from concept_audit.models import NativeCBM, NativeCEM
from concept_audit.readouts import CoordinateReadout, GroupReadout, IdentityReadout
import numpy as np
from experiments.semantic_edits.native_adapter import (load_student, encode_features, cached_latents, readout_and_logits,
                                                       decode_levels, decode_calibrated)
from experiments.semantic_edits.torch_operators import AffineEdit, MLPEdit, fit_operator


def save_like_train_and_audit(model, backend, path, seed=0):
    # Same keys as concept_audit.training.train_and_audit's torch.save call.
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "backend": backend,
                "feature_dim": 6, "num_classes": 5, "blocks": model.blocks,
                "readout_matrix": model.readout.matrix.cpu(), "transform_matrix": torch.eye(model.latent_dim()),
                "seed": seed}, path)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.k, self.z = 3, torch.randn(20, 6)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def models(self):
        k = self.k
        yield "coordinates", "native", NativeCBM(6, 5, CoordinateReadout(2*k, range(k)), [(j, k+j) for j in range(k)])
        yield "identity", "native", NativeCBM(6, 5, IdentityReadout(k))
        blocks = [(2*j, 2*j+1) for j in range(k)]
        yield "grouped", "native", NativeCBM(6, 5, GroupReadout(2*k, blocks), blocks)
        yield "block_projection", "cem", NativeCEM(6, 5, k)

    def test_rebuild_is_frozen_and_forward_equivalent(self):
        for name, backend, model in self.models():
            path = Path(self.dir.name) / f"{name}.pt"
            save_like_train_and_audit(model.eval(), backend, path)
            loaded, readout_name, _ = load_student(path, dtype=torch.float32)
            self.assertEqual(readout_name, name)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            with torch.no_grad():
                for a, b in zip(model(self.z), loaded(self.z)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mismatched_readout_is_rejected(self):
        _, _, model = next(self.models())
        path = Path(self.dir.name) / "bad.pt"
        save_like_train_and_audit(model, "native", path)
        ckpt = torch.load(path, weights_only=True)
        ckpt["readout_matrix"] = ckpt["readout_matrix"].flip(1)
        torch.save(ckpt, path)
        with self.assertRaises(ValueError):
            load_student(path)

    def test_cached_encoding_reproduces_forward_and_is_hash_bound(self):
        for name, backend, model in self.models():
            path = Path(self.dir.name) / f"{name}.pt"
            save_like_train_and_audit(model.eval(), backend, path)
            loaded, _, _ = load_student(path)
            cache = Path(self.dir.name) / f"cache_{name}"
            cache.mkdir()
            np.save(cache / "embeddings.npy", self.z.numpy())
            np.save(cache / "ids.npy", np.arange(len(self.z)).astype(str))
            c = cached_latents(loaded, path, cache, Path(self.dir.name) / f"lat_{name}.npz", "cpu")
            torch.testing.assert_close(c, encode_features(loaded, self.z.numpy(), "cpu", batch=7))
            scores, logits = readout_and_logits(loaded, c, "cpu", batch=6)
            ref_scores, ref_logits = loaded(self.z.double())
            torch.testing.assert_close(scores, ref_scores)
            torch.testing.assert_close(logits, ref_logits)
            other = Path(self.dir.name) / f"other_{name}.pt"
            save_like_train_and_audit(model.eval(), backend, other, seed=1)
            with self.assertRaises(ValueError):  # same cache file, different checkpoint bytes
                cached_latents(loaded, other, cache, Path(self.dir.name) / f"lat_{name}.npz", "cpu")

    def test_readout_units_decode_and_logit_intervention_round_trips(self):
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        scores = torch.logit(torch.tensor([0.01, 0.26, 0.49, 0.8, 0.999]))
        self.assertEqual(decode_levels(scores, levels).tolist(), [0, 1, 2, 3, 4])
        ref = [[-3., -2.], [0., 0.], [3., 2.]]
        self.assertEqual(decode_calibrated(torch.tensor([[-2.5, -2.], [0.4, -.1], [2., 3.]]), ref).tolist(), [0, 1, 2])
        _, _, model = next(self.models())
        c = model.eval().encode(self.z)
        target = torch.logit(torch.tensor(0.75))
        edited = model.intervene(c, 1, target)
        self.assertEqual(decode_levels(model.concept_readout(edited)[:, 1], levels).unique().tolist(), [3])
        torch.testing.assert_close(model.concept_readout(edited)[:, [0, 2]], model.concept_readout(c)[:, [0, 2]])

    def test_operators_start_at_identity_and_ignore_target_at_inference(self):
        x, a = torch.randn(30, 4), torch.nn.functional.one_hot(torch.arange(30) % 3, 3).float()
        for op in (AffineEdit(4, 3), MLPEdit(4, 3)):
            torch.testing.assert_close(op(x, a), x)
        y = x + a @ torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        op, info = fit_operator("affine", x, a, y, (x, a, y), seed=0, steps=300, batch=30, lr=0.05, eval_every=50,
                                device="cpu")
        self.assertLess(info["selected_val_mse"], 1e-3)
        self.assertFalse(any(p.requires_grad for p in op.parameters()))
        self.assertEqual(op.forward.__code__.co_argcount, 3)  # (self, x, action): no target-image argument


if __name__ == "__main__":
    unittest.main()
