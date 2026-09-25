import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_resource.q1.ab_experiment import (
    ATOMIC_FIELDS, DSIR_FIELDS, Q_FIELDS, build_mappings,
    candidate_conflict_edges, fit_dsir_rank_reference, transform_dsir_rank,
)
from llm_resource.q1.schema import RAW_COLUMNS
from llm_resource.q1.scoring import interaction_weights, quality


class ABExperimentTests(unittest.TestCase):
    def test_concept_counts_and_exact_atomic_coverage(self):
        mappings = build_mappings(np.array([.1, .2, .3, .4]), np.full(4, .25))
        self.assertEqual([len(mappings[k][0]) for k in mappings], [22, 23, 20])
        for names, matrix in mappings.values():
            self.assertEqual(matrix.shape, (25, len(names)))
            np.testing.assert_array_equal((matrix > 0).sum(axis=1), np.ones(25))
        a_names, a = mappings["A22"]
        q_idx = [ATOMIC_FIELDS.index(n) for n in Q_FIELDS]
        np.testing.assert_allclose(a[q_idx, a_names.index("qurater")], np.full(4, .25))
        for scheme in ("B23_flat", "B20_hierarchical"):
            names, mapping = mappings[scheme]
            dsir_column = next(j for j, n in enumerate(names) if n.startswith("dsir_target"))
            np.testing.assert_allclose(
                mapping[[ATOMIC_FIELDS.index(n) for n in DSIR_FIELDS], dsir_column],
                np.full(3, 1 / 3),
            )

    def test_conflict_pair_expansion_preserves_monotonicity(self):
        pairs = [["fineweb_edu", "ad_en"], ["qurater", "ad_en"]]
        for names, _ in build_mappings(np.full(4, .25), np.full(4, .25)).values():
            edges = candidate_conflict_edges(names, pairs)
            self.assertEqual(len(edges), 5 if len(names) == 23 else 2)
            weights = np.full(len(names), 1 / len(names))
            theta = interaction_weights(weights, edges, .25)
            x = np.full((1, len(names)), .4)
            for j in range(len(names)):
                y = x.copy()
                y[0, j] += .1
                self.assertGreaterEqual(quality(y, weights, edges, theta)[1][0],
                                        quality(x, weights, edges, theta)[1][0] - 1e-12)

    def test_dsir_rank_is_frozen_and_monotone(self):
        raw = np.zeros((8, len(RAW_COLUMNS)))
        for name in DSIR_FIELDS:
            raw[:, RAW_COLUMNS.index(name)] = np.arange(8)
        domains = np.array(["x"] * 4 + ["y"] * 4)
        reference = fit_dsir_rank_reference(raw, domains, 11)
        original = np.full_like(raw, .5)
        preprocessor = {"domain_medians": {"x": [0.] * raw.shape[1], "y": [0.] * raw.shape[1]},
                        "global_medians": [0.] * raw.shape[1]}
        test = raw.copy()
        test[0, RAW_COLUMNS.index(DSIR_FIELDS[0])] = np.nan
        result = transform_dsir_rank(test, domains, preprocessor, reference, original)
        j = RAW_COLUMNS.index(DSIR_FIELDS[1])
        self.assertTrue(np.all(np.diff(result[:, j]) >= 0))
        self.assertGreater(result[-1, j], result[0, j])
        self.assertTrue(np.isfinite(result).all())
        unchanged = [j for j, name in enumerate(RAW_COLUMNS) if name not in DSIR_FIELDS]
        np.testing.assert_allclose(result[:, unchanged], original[:, unchanged])


if __name__ == "__main__":
    unittest.main()
