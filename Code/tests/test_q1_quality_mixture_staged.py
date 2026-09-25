import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_resource.q1.mixture import _ridge_fit, _ridge_predict
from llm_resource.q1.quality_mixture_staged import (
    _prior_ridge_fit, _prior_ridge_predict, complete_a16_mapping, quality_geometry,
)


class StagedQualityMixtureTests(unittest.TestCase):
    def test_completed_a16_quality_uses_all_seventeen_domains(self):
        p = np.tile(np.arange(1, 18, dtype=float) / 153, (3, 1))
        q = np.arange(17, dtype=float)
        z = quality_geometry(p, "M3-A16", q, 1.0)
        np.testing.assert_allclose(z[:, 17:], p * q)
        self.assertTrue(np.any(z[:, 17 + 6:] != 0))

    def test_completed_mapping_keeps_teacher_links(self):
        domains = [f"domain_{i}" for i in range(17)]
        a1 = [f"quality_{i}" for i in range(7)]
        sim = pd.DataFrame([{
            "domain": domain, "a16_known": i < 6,
            "a16_mapping_type": "direct" if i < 6 else "inferred",
            "a16_quality_domain": a1[i] if i < 6 else "(none)",
            "a16_reference_q": .1 * (i + 1) if i < 6 else np.nan,
            "top1_quality_domain": a1[6], "top2_quality_domain": a1[0],
            "top1_probability": .4, "distance_margin": .5,
            "q_similarity": .4, "q_sensitivity_min": .3,
            "q_sensitivity_max": .5, "a18_rows": 64,
        } for i, domain in enumerate(domains)])
        sensitivity = pd.DataFrame([{"domain": domain, "omitted_field": "signal_1",
                                     "temperature": 1.0, "top1_quality_domain": a1[6]}
                                    for domain in domains])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            pd.DataFrame({"dataset": ["A1"] * 7, "domain": a1,
                          "Q_primary_mean": [.1 * (i + 1) for i in range(7)]}).to_csv(path, index=False)
            result = complete_a16_mapping(sim, sensitivity, path)
        self.assertEqual(result.assignment_source.eq("teacher_provided").sum(), 6)
        self.assertEqual(result.assignment_source.eq("nine_signal_top1_inference").sum(), 11)
        self.assertEqual(result.loc[0, "assigned_quality_domain"], a1[0])
        self.assertEqual(result.loc[6, "assigned_quality_domain"], a1[6])

    def test_zero_quality_prior_is_existing_quadratic_ridge(self):
        rng = np.random.default_rng(19)
        p = rng.dirichlet(np.ones(17), size=48)
        y = rng.normal(size=(48, 3))
        q = rng.normal(size=17)
        ours = _prior_ridge_fit(p, y, q, np.ones(17), .01, 10.0, 0.0)
        existing = _ridge_fit(p, y, .01, 10.0, quadratic=True)
        np.testing.assert_allclose(_prior_ridge_predict(ours, p),
                                   _ridge_predict(existing, p), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
