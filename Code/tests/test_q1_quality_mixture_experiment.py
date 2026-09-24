import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_resource.q1.quality_mixture_experiment import centered_quality, geometry, score
from llm_resource.q1.reconstructed_signals import RPS_FIELDS, rps_signals


class QualityMixtureExperimentTests(unittest.TestCase):
    def test_rps_reconstruction_has_all_eleven_fields(self):
        repeated = rps_signals("A b a b a b.")
        diverse = rps_signals("A b c d e f.")
        self.assertEqual(set(repeated), set(RPS_FIELDS))
        self.assertEqual(repeated["rps_doc_word_count"], 6)
        self.assertGreater(repeated["rps_doc_frac_chars_top_2gram"],
                           diverse["rps_doc_frac_chars_top_2gram"])
        self.assertEqual(rps_signals("A1!")["rps_doc_frac_no_alph_words"], 100 * 2 / 3)

    def test_fixed_quality_expansion_does_not_add_linear_information(self):
        rng = np.random.default_rng(4)
        p = rng.dirichlet(np.ones(17), size=64)
        q = rng.normal(size=17)
        base = geometry(p, 1, q)
        for model in (2, 3):
            expanded = geometry(p, model, q)
            self.assertEqual(np.linalg.matrix_rank(base), np.linalg.matrix_rank(expanded))

    def test_two_quality_paths_have_same_feature_shape(self):
        p = np.full((3, 17), 1 / 17)
        q1 = centered_quality(np.arange(17, dtype=float))
        q2 = centered_quality(np.arange(17, dtype=float)[::-1])
        self.assertEqual(geometry(p, 2, q1).shape, geometry(p, 3, q2).shape)

    def test_score_recovers_perfect_prediction(self):
        actual = np.arange(100, dtype=float).reshape(20, 5)
        result = score(actual, actual.copy())
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["r2"], 1.0)
        self.assertEqual(result["top10_overlap"], 1.0)
        self.assertEqual(result["regret_at_1"], 0.0)
        self.assertEqual(result["regret_at_5"], 0.0)
        self.assertEqual(result["regret_at_10"], 0.0)


if __name__ == "__main__":
    unittest.main()
