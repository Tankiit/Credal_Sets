import unittest
import numpy as np
from experiments.semantic_edits.build_pairs import make_pairs, validate_pairs, heldout_combination, SPLITS


def factorial(cards):
    grids = np.meshgrid(*[np.arange(c) for c in cards.values()], indexing='ij')
    return {f: g.ravel().astype(np.int64) for f, g in zip(cards, grids)}


class PairTests(unittest.TestCase):
    def setUp(self):
        self.f = factorial({'floor': 5, 'wall': 3, 'shape': 4, 'size': 2})
        self.labels = self.f['shape'] * 10 + self.f['wall']
        self.pairs, self.info = make_pairs(self.f, 'floor', 'shape', holdout=1, holdout_step=2, pairs_per_context=4, seed=1)

    def test_valid_pairs_pass_all_checks(self):
        checks = validate_pairs(self.pairs, self.f, 'floor', self.info, hashes=np.arange(len(self.labels)).astype(str),
                                labels=self.labels)
        self.assertTrue(all(checks.values()))
        self.assertEqual(set(np.unique(self.pairs['split'])), {0, 1, 2})

    def test_only_edit_factor_changes(self):
        s, t = self.pairs['source'], self.pairs['target']
        self.assertTrue((self.f['floor'][s] != self.f['floor'][t]).all())
        for name in ('wall', 'shape', 'size'):
            self.assertTrue((self.f[name][s] == self.f[name][t]).all())

    def test_context_keeps_all_edit_values_in_one_split(self):
        for c in np.unique(self.info['context']):
            self.assertEqual(len(set(self.info['image_split'][self.info['context'] == c])), 1)

    def test_heldout_combinations_only_in_test(self):
        fit = self.pairs['split'] < 2
        self.assertFalse(self.pairs['target_heldout_combination'][fit].any())
        self.assertFalse(self.pairs['source_heldout_combination'][fit].any())
        self.assertTrue(self.pairs['target_heldout_combination'][~fit].any())
        # every edit value still appears in operator training with some shape
        train = self.pairs['split'] == 0
        self.assertEqual(set(self.pairs['target_level'][train]), set(range(5)))

    def test_heldout_rule_holds_out_conjunctions_not_values(self):
        held = heldout_combination(np.arange(5)[:, None], np.arange(4)[None], 5, 1, 2)
        self.assertTrue(held.any() and (~held).any(1).all())  # some conjunctions held out; every value seen
        self.assertTrue((~held).any(0).all())

    def test_corrupted_pairs_are_rejected(self):
        bad = {k: v.copy() for k, v in self.pairs.items()}
        i = int(np.flatnonzero(bad['split'] == 0)[0])
        bad['target'][i] = bad['source'][i] + 1  # changes wall/size or keeps floor
        with self.assertRaises(AssertionError):
            validate_pairs(bad, self.f, 'floor', self.info)
        leak = {k: v.copy() for k, v in self.pairs.items()}
        tr, te = np.flatnonzero(leak['split'] == 0)[0], np.flatnonzero(leak['split'] == 2)[0]
        leak['split'][te] = 0  # a test-context pair relabelled as training
        with self.assertRaises(AssertionError):
            validate_pairs(leak, self.f, 'floor', self.info)
        dup = np.arange(len(self.labels)).astype(str)
        a = np.flatnonzero(self.info['image_split'] == SPLITS[0])
        b = np.flatnonzero(self.info['image_split'] == SPLITS[2])
        used_a = np.intersect1d(a, self.pairs['source'][self.pairs['split'] == 0])[0]
        used_b = np.intersect1d(b, self.pairs['source'][self.pairs['split'] == 2])[0]
        dup[used_b] = dup[used_a]  # identical image bytes under two ids in different splits
        with self.assertRaises(AssertionError):
            validate_pairs(self.pairs, self.f, 'floor', self.info, hashes=dup)


if __name__ == '__main__':
    unittest.main()
