import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from llm_resource.q1.quality_mixture_14 import (
    apply_proxy_calibration, fit_proxy_calibration, similarity_weights,
    score_b20, validate_signal_config,
)
from llm_resource.q1.quality_mixture_14_loss import geometry
from llm_resource.q1.schema import RAW_COLUMNS


class FourteenSignalStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((CODE / "configs/q1_quality_mixture_14_stage.json").read_text(encoding="utf-8"))

    def test_distance_budget_is_equal_across_five_families(self):
        fields = self.config["accepted_fields_required"]
        weights = validate_signal_config(fields, self.config)
        self.assertAlmostEqual(weights.sum(), 1)
        for members in self.config["distance_families"].values():
            self.assertAlmostEqual(sum(weights[fields.index(name)] for name in members), .2)

    def test_similarity_rows_normalize_and_honor_temperature(self):
        components = np.array([[[1., 0.], [0., 1.]]])
        distance, weights = similarity_weights(components, np.array([.75, .25]), 1.)
        self.assertEqual(distance.shape, (1, 2))
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertGreater(weights[0, 1], weights[0, 0])
        _, cooler = similarity_weights(components, np.array([.75, .25]), .5)
        self.assertGreater(cooler[0, 1], weights[0, 1])

    def test_proxy_calibration_has_nonnegative_slope_and_clips(self):
        fields = ["fineweb_edu", "modernbert_readability"]
        official = np.zeros((30, len(RAW_COLUMNS)))
        rebuilt = official.copy()
        for name in fields:
            j = RAW_COLUMNS.index(name)
            rebuilt[:, j] = np.linspace(0, 1, 30)
            official[:, j] = 1 - rebuilt[:, j]
        fit = fit_proxy_calibration(official, rebuilt, fields)
        self.assertEqual(fit["fineweb_edu"]["slope"], 0)
        self.assertEqual(fit["modernbert_readability"]["slope"], 0)
        transformed = apply_proxy_calibration(rebuilt, fit)
        self.assertTrue(np.all((transformed[:, RAW_COLUMNS.index("modernbert_readability")] >= 0) &
                               (transformed[:, RAW_COLUMNS.index("modernbert_readability")] <= 1)))

    def test_zero_quality_block_retains_p_distances(self):
        p = np.array([[.2, .8], [.7, .3]])
        q = np.array([-1., 1.])
        reference = np.sum((p[0] - p[1]) ** 2)
        for model in ("M2-bridge", "M3-direct"):
            z = geometry(p, model, q, 0.)
            self.assertAlmostEqual(np.sum((z[0] - z[1]) ** 2), reference)

    def test_cross_corpus_missing_uses_global_medians_but_keeps_real_domains(self):
        raw = np.array([[np.nan, 2.], [3., np.nan]])
        domains = np.array(['arxiv', 'github'])
        fit = {'global_medians': [1., 4.]}
        with patch('llm_resource.q1.quality_mixture_14.transform') as transform:
            transform.side_effect = lambda values, names, _: (None, None, None, values)
            result = score_b20(raw, domains, fit, np.eye(2), np.array([.5, .5]),
                               global_missing=True)
        self.assertTrue(np.array_equal(transform.call_args.args[1], domains))
        self.assertTrue(np.array_equal(transform.call_args.args[0], np.array([[1., 2.], [3., 4.]])))
        self.assertTrue(np.allclose(result, [1.5, 3.5]))


if __name__ == "__main__":
    unittest.main()
