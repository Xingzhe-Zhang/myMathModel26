import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))

from llm_resource.q1.dimension_evaluation import evaluate_independent_labels  # noqa: E402
from llm_resource.q1.dimensions import ATOMIC_FIELDS, score_dimension_views, validate_dimension_config  # noqa: E402
from llm_resource.q1.schema import FIELDS, RAW_COLUMNS  # noqa: E402


class DimensionSchemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((CODE / "configs/q1_dimension_schemes.json").read_text(encoding="utf-8"))

    def test_all_atomic_signals_are_partitioned_and_weights_sum_to_one(self):
        validate_dimension_config(self.config)
        self.assertEqual(len(ATOMIC_FIELDS), 25)
        z25 = np.random.default_rng(3).uniform(size=(10, 25))
        z22 = np.column_stack([
            z25[:, [RAW_COLUMNS.index(f"qurater_{i}") for i in range(4)]].mean(axis=1)
            if field == "qurater" else z25[:, RAW_COLUMNS.index(field)] for field in FIELDS
        ])
        scores, weights = score_dimension_views(z22, z25, self.config)
        np.testing.assert_allclose(scores.Q22_equal, z22.mean(axis=1))
        np.testing.assert_allclose(scores.Q_semantic, z25 @ weights.Q_semantic_weight.to_numpy())
        self.assertAlmostEqual(weights.Q22_equal_weight.sum(), 1)
        self.assertAlmostEqual(weights.Q_semantic_weight.sum(), 1)
        self.assertAlmostEqual(weights.loc[weights.atomic_field.str.startswith("qurater_"), "Q22_equal_weight"].sum(), 1/22)
        self.assertAlmostEqual(weights.loc[weights.atomic_field.str.startswith("qurater_"), "Q_semantic_weight"].sum(), 1/22)
        self.assertAlmostEqual(weights.loc[weights.atomic_field == "fluency_en", "Q_semantic_weight"].iloc[0], 21/220)
        self.assertTrue(scores.Q_semantic.between(0, 1).all())

    def test_duplicate_signal_is_rejected(self):
        config = deepcopy(self.config)
        config["groups"]["expression"].append(["dsir_books"])
        with self.assertRaises(ValueError):
            validate_dimension_config(config)

    def test_sparse_labels_cannot_select_a_winner(self):
        scores = pd.DataFrame({"uid": ["a", "b"], "domain": ["book", "book"],
                               "Q22_equal": [.5, .7], "Q_semantic": [.6, .8],
                               "is_A1_reference": [True, True]})
        labels = pd.DataFrame({"uid": ["a", "b"], "quality": [.2, .8], "defect": [1, 0]})
        result, _ = evaluate_independent_labels(scores, labels, self.config, 3)
        self.assertEqual(result["decision"], "insufficient_evidence")

    def test_independent_labels_can_select_clear_winner(self):
        config = deepcopy(self.config)
        config["human_evaluation"]["bootstrap_repeats"] = 100
        rows = []
        for domain in ("arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"):
            for i in range(50):
                quality = i / 49
                rows.append({"uid": f"{domain}-{i}", "domain": domain,
                             "Q22_equal": 1-quality, "Q_semantic": quality,
                             "quality": quality, "defect": int(quality < .5),
                             "is_A1_reference": True})
        frame = pd.DataFrame(rows)
        result, _ = evaluate_independent_labels(
            frame.drop(columns=["quality", "defect"]), frame[["uid", "quality", "defect"]], config, 3
        )
        self.assertEqual(result["decision"], "semantic_supported_on_this_independent_label_set")


if __name__ == "__main__":
    unittest.main()
