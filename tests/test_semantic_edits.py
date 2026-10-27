import unittest
import numpy as np
from experiments.semantic_edits.data import scenes,pairs,concepts,heldout,render,edit,action
from experiments.semantic_edits.models import coordinate, Ridge
from experiments.semantic_edits.metrics import evaluate

class ProtocolTests(unittest.TestCase):
    def test_edit_changes_only_requested_attribute(self):
        for s,t,o,c in pairs(scenes(30,'test',0)):
            changed = np.flatnonzero(concepts(s)!=concepts(t))
            self.assertTrue(all(5*o <= k < 5*o+3 for k in changed))
            self.assertEqual(s.offsets,t.offsets)
            self.assertEqual(s.shapes,t.shapes)
            self.assertEqual(s.scene_id,t.scene_id)
    def test_holdout_both_endpoints(self):
        for split in ('train','val'):
            self.assertTrue(all(not heldout(s) and not heldout(t) for s,t,_,_ in pairs(scenes(20,split,0))))
    def test_coordinate_oracle_and_residual_preservation(self):
        s,t,o,c = pairs(scenes(1,'test',0))[0]
        z = np.r_[concepts(s),[.2,.9]][None]
        out = coordinate(z,action(o,c)[None])
        np.testing.assert_array_equal(out[0,:10],concepts(t))
        np.testing.assert_array_equal(out[0,10:],z[0,10:])
    def test_identity_cannot_win_edit_success(self):
        pp = pairs(scenes(20,'test',3))
        src = np.stack([concepts(s) for s,_,_,_ in pp])
        dst = np.stack([concepts(t) for _,t,_,_ in pp])
        m = evaluate(src,src,dst,dst,[o for _,_,o,_ in pp])
        self.assertFalse(m['target_success'].any())
        self.assertTrue((m['unaffected_accuracy']==1).all())
        m = evaluate(src,dst,dst,dst,[o for _,_,o,_ in pp])
        self.assertTrue(m['joint_success'].all())
    def test_ridge_affine_map(self):
        x = np.random.default_rng(1).normal(size=(100,4)); y=x[:,:2]*2+3
        np.testing.assert_allclose(Ridge(1e-8).fit(x,y)(x),y,atol=1e-7)
    def test_deterministic_render(self):
        self.assertEqual(render(scenes(1,'test',0)[0]).tobytes(),render(scenes(1,'test',0)[0]).tobytes())

if __name__ == '__main__': unittest.main()
