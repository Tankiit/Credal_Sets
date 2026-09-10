import importlib.util
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import builtins
from dataclasses import replace
import torch
from concept_audit.models import NativeCBM, NativeCEM, PyCAdapter
from concept_audit.readouts import IdentityReadout, CoordinateReadout, GroupReadout, LinearReadout, BlockProjectionReadout
from concept_audit.transforms import admissible_transform, ReparameterizedModel
from concept_audit.diagnostics import AuditState, default_registry, TaskProbe
from concept_audit.audits import audit_equivalence, audit_structural, audit_informational, audit_consequence


class AuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.z = torch.randn(80, 5, dtype=torch.double)
        self.g = (self.z[:, :2] > 0).double()
        self.y = (self.z[:, 2] > 0).long()
        self.train = torch.arange(80) < 50

    def state(self, model):
        model = model.double().eval()
        return AuditState(model.encode(self.z).detach(), self.g, self.y, model, self.train)

    def test_all_readouts_exact_equivalence(self):
        for r in (IdentityReadout(2), CoordinateReadout(4, [0, 1]),
                  GroupReadout(4, [[0, 1], [2, 3]]),
                  BlockProjectionReadout([[1., 2.], [3., 1.]]),
                  LinearReadout([[1., 2., 0., 1.], [2., 4., 0., 2.]])):
            with self.subTest(readout=type(r).__name__):
                state = self.state(NativeCBM(5, 2, r))
                a = admissible_transform(state.model.readout)
                report = audit_equivalence(state, a, default_registry(), tol=1e-9)
                self.assertLess(report['max_logit_error'], 1e-9)
                transformed = ReparameterizedModel(state.model, a)
                torch.testing.assert_close(transformed.encode(self.z), state.c @ a.T)
                if isinstance(r, IdentityReadout):
                    torch.testing.assert_close(a, torch.eye(2, dtype=torch.double))

    def test_reject_invalid_transforms(self):
        model = NativeCBM(5, 2, CoordinateReadout(3, [0, 1])).double()
        for a in (torch.eye(3)*2, torch.diag(torch.tensor([1., 1., 0.])), torch.full((3,3), float('nan'))):
            with self.assertRaises(ValueError):
                ReparameterizedModel(model, a)

    def test_compensated_bias_and_transport(self):
        state = self.state(NativeCBM(5, 2, CoordinateReadout(4, [0,1]), [(0,2),(1,3)]))
        a = admissible_transform(state.model.readout, strength=1.)
        changed = ReparameterizedModel(state.model, a)
        c2 = state.c @ a.T
        donor = state.c.roll(1, 0)
        torch.testing.assert_close(changed.head.bias, state.model.head.bias)
        for j in range(2):
            old = state.model.intervene(state.c, j, 2.)
            torch.testing.assert_close(changed.intervene(c2, j, 2.), old @ a.T)
            expected = state.model.substitute_donor(state.c, j, donor) @ a.T
            torch.testing.assert_close(changed.substitute_donor(c2, j, donor @ a.T), expected)
        left = audit_consequence(state, default_registry())
        right = audit_consequence(replace(state, c=c2, model=changed), default_registry())
        self.assertEqual([r['accuracy_drop'] for r in left['per_concept']], [r['accuracy_drop'] for r in right['per_concept']])

    def test_intervention_preserves_other_scores_and_input(self):
        state = self.state(NativeCEM(5, 2, 2, 3))
        original = state.c.clone()
        changed = state.model.intervene(state.c, 1, torch.ones(80))
        torch.testing.assert_close(state.c, original)
        torch.testing.assert_close(state.model.concept_readout(changed)[:,0], state.model.concept_readout(original)[:,0])
        torch.testing.assert_close(state.model.concept_readout(changed)[:,1], torch.ones(80, dtype=torch.double))
        model = NativeCBM(5, 2, LinearReadout([[1., 0.], [1., 0.]])).double()
        with self.assertRaises(ValueError):
            model.intervene(torch.zeros(3,2,dtype=torch.double), 0, 1.)

    def test_audits_do_not_mutate_and_are_reproducible(self):
        state = self.state(NativeCBM(5, 2, CoordinateReadout(4,[0,1])))
        c, w = state.c.clone(), state.model.head.weight.clone()
        registry = default_registry()
        audit_structural(state, registry)
        self.assertEqual(audit_informational(state, registry, 9), audit_informational(state, registry, 9))
        audit_consequence(state, registry)
        torch.testing.assert_close(state.c, c)
        torch.testing.assert_close(state.model.head.weight, w)

    def test_probe_scores_only_held_out_samples(self):
        state = self.state(NativeCBM(5, 2, IdentityReadout(2)))
        probe = TaskProbe()
        accuracy = probe.compute(state)
        labels = state.labels.clone()
        labels[~state.train_mask] = 1-labels[~state.train_mask]
        torch.testing.assert_close(probe.compute(replace(state, labels=labels)), 1-accuracy)

    def test_native_pipeline_without_optional_imports(self):
        original_import = builtins.__import__
        def core_only(name, *args, **kwargs):
            if name.split('.')[0] in {'torch_concepts', 'probly'}:
                raise ImportError('Optional backend deliberately unavailable')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=core_only):
            state = self.state(NativeCBM(5, 2, CoordinateReadout(4, [0,1])))
            audit_equivalence(state, admissible_transform(state.model.readout), default_registry())
            audit_consequence(state, default_registry())

    def test_dataset_independent_cache_alignment(self):
        from concept_audit.data.extraction import extract_cache
        from concept_audit.data import load_cache
        from models.backbones import BackboneInfo
        class FakeBackbone:
            info = BackboneInfo('test', 'fake', 2)
            def encode_pil(self, images):
                return torch.tensor([[float(i), -float(i)] for i in images])
        dataset = [(i, [i % 2, (i+1) % 2], i % 3) for i in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            extract_cache(dataset, FakeBackbone(), directory, batch_size=3)
            z, g, y = load_cache(directory)
            torch.testing.assert_close(z[:,0], torch.arange(7).float())
            self.assertEqual(y.tolist(), [i % 3 for i in range(7)])
            self.assertEqual(g[:,0].tolist(), [i % 2 for i in range(7)])
            self.assertTrue((Path(directory) / 'meta.json').is_file())

    @unittest.skipUnless(importlib.util.find_spec('torch_concepts'), 'optional PyC is not installed')
    def test_real_pyc_backend_same_audits(self):
        native = NativeCBM(5, 2, CoordinateReadout(4,[0,1]), [(0,2),(1,3)]).double()
        pyc = PyCAdapter(5, 2, CoordinateReadout(4,[0,1]), [(0,2),(1,3)]).double()
        pyc.encoder.encoder.load_state_dict(native.encoder.state_dict())
        pyc.head.load_state_dict(native.head.state_dict())
        registry = default_registry()
        left, right = self.state(native), self.state(pyc)
        torch.testing.assert_close(left.c, right.c)
        a = admissible_transform(native.readout)
        self.assertEqual(audit_equivalence(left,a,registry), audit_equivalence(right,a,registry))
        self.assertEqual(audit_structural(left,registry), audit_structural(right,registry))
        self.assertEqual(audit_informational(left,registry), audit_informational(right,registry))
        self.assertEqual(audit_consequence(left,registry), audit_consequence(right,registry))
        for model in (native, pyc):
            model.zero_grad()
            scores, logits = model(self.z)
            (scores.square().mean()+logits.square().mean()).backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
