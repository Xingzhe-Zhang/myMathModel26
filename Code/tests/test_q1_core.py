import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_resource.q1.schema import decode_signals
from llm_resource.q1.scoring import interaction_weights, quality
from llm_resource.q1.schema import FIELDS


class Q1CoreTests(unittest.TestCase):
    def test_decoding_direction(self):
        row = {f: 0.5 for f in FIELDS}
        row.update({
            "fineweb_edu": [2.5],
            "fluency_en": [0.0, 10.0],
            "ad_en": [10.0, 0.0],
            "qurater": [1.0, 2.0, 3.0, 4.0],
        })
        for name in ["modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning", "modernbert_professionalism"]:
            row[name] = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        values, _, issues = decode_signals(row)
        self.assertFalse(issues)
        self.assertGreater(values[1], 0.99)
        # four QuRating subscales are expanded before ad_en in the raw vector
        self.assertLess(values[10], 0.01)

    def test_finite_compensation_is_monotone(self):
        w = np.full(22, 1 / 22)
        edges = [(0, 1), (2, 3)]
        theta = interaction_weights(w, edges, 0.5)
        x = np.full((1, 22), 0.4)
        y = x.copy()
        y[0, 0] += 0.1
        qx = quality(x, w, edges, theta)[1][0]
        qy = quality(y, w, edges, theta)[1][0]
        self.assertGreaterEqual(qy, qx)
        self.assertGreaterEqual(qx, 0.0)
        self.assertLessEqual(qx, 1.0)

    def test_invalid_interaction_is_rejected(self):
        w = np.full(22, 1 / 22)
        with self.assertRaises(ValueError):
            interaction_weights(w, [(0, 1), (0, 2)], 2.0)


if __name__ == "__main__":
    unittest.main()
