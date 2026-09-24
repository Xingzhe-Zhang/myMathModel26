import unittest
import sys
import json
import lzma
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_resource.q1.mixture import grouped_folds, mixture_features, evaluate, quality_bridge
from llm_resource.q1.quality_evidence import audit_regmix_text


class MixtureCoreTests(unittest.TestCase):
    def test_quadratic_scheffe_has_153_features(self):
        p = np.full((3, 17), 1 / 17)
        self.assertEqual(mixture_features(p, True).shape, (3, 153))

    def test_grouped_folds_keep_near_duplicates_together(self):
        p = np.zeros((4, 3), float)
        p[0] = [1, 0, 0]
        p[1] = [.999, .001, 0]
        p[2] = [0, 1, 0]
        p[3] = [0, 0, 1]
        folds, groups = grouped_folds(p, .01, 2, 7)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(folds[0], folds[1])

    def test_evaluation_regret_is_nonnegative(self):
        actual = np.array([[1., 2.], [2., 1.]])
        predicted = np.array([[1.5, 1.5], [1.2, 1.8]])
        self.assertGreaterEqual(evaluate(actual, predicted)["selected_regret"], 0)

    def test_a16_a18_bridge_preserves_missing_quality_and_audits_text(self):
        domains = ["arxiv"] + [f"domain_{i}" for i in range(16)]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a16, a17, a18, scores = (root / name for name in
                                      ("a16.csv", "a17.csv", "a18.jsonl.xz", "scores.csv"))
            pd.DataFrame([{"mixture_domain": d, "quality_domain": "arxiv" if i == 0 else "(none)",
                           "mapping_type": "direct" if i == 0 else "inferred"}
                          for i, d in enumerate(domains)]).to_csv(a16, index=False)
            pd.DataFrame([{"domain": d, "source_path": f"valid/{d}.jsonl", "sample_rows": 1,
                           "sample_bytes_requested": 100, "avg_text_chars": 4.0}
                          for d in domains]).to_csv(a17, index=False)
            with lzma.open(a18, "wt", encoding="utf-8") as target:
                for d in domains:
                    target.write(json.dumps({"text": "abcd", "_source_domain": d,
                                             "_source_path": f"valid/{d}.jsonl"}) + "\n")
            pd.DataFrame([{"dataset": "A1", "domain": "arxiv", "Q_primary_mean": .8,
                           "ci_low_primary": .75, "ci_high_primary": .85},
                          {"dataset": "A2", "domain": "arxiv", "Q_primary_mean": .7}]).to_csv(scores, index=False)
            evidence, meta = audit_regmix_text(a16, a17, a18, domains)
            self.assertEqual(meta["a18_total_rows"], 17)
            self.assertTrue(meta["a17_a18_row_path_mean_reconciled"])
            self.assertTrue(evidence["a18_source_is_valid_shard"].all())
            p = np.array([[.25] + [.75] + [0.] * 15])
            domain, recipe, bridge_meta, _ = quality_bridge(scores, a16, a17, a18,
                                                              p, domains, np.array([1]))
            self.assertEqual(len(domain), 17)
            self.assertEqual(bridge_meta["mapped_domains"], 1)
            self.assertAlmostEqual(recipe.loc[0, "mapped_mass"], .25)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_on_mapped_mass"], .8)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_lower_if_unmapped_zero"], .2)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_upper_if_unmapped_one"], .95)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_lower_with_a1_ci"], .1875)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_upper_with_a1_ci"], .9625)
            self.assertAlmostEqual(recipe.loc[0, "q_mix_extension_sensitivity"], .775)


if __name__ == "__main__":
    unittest.main()
