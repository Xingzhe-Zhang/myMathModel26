import json
import lzma
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_resource.q1.quality_reconstruction import _sample


class QualitySamplingTests(unittest.TestCase):
    def test_underfilled_a18_domain_uses_available_rows_and_reports_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl.xz"
            records = [
                {"_source_domain": domain, "_source_path": f"valid/{domain}/{i}", "text": f"text {i}"}
                for domain, size in (("large", 4), ("ubuntu_irc", 2))
                for i in range(size)
            ]
            with lzma.open(path, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            coverage = {}
            sampled = _sample(path, "text", 3, {"large", "ubuntu_irc"},
                              allow_shortfall=True, coverage=coverage)
            self.assertEqual(len(sampled), 5)
            self.assertEqual(coverage["large"], {"available": 4, "selected": 3, "target": 3})
            self.assertEqual(coverage["ubuntu_irc"], {"available": 2, "selected": 2, "target": 3})
            self.assertEqual(sampled, _sample(path, "text", 3, {"large", "ubuntu_irc"},
                                               allow_shortfall=True))
            with self.assertRaisesRegex(ValueError, "ubuntu_irc.*2"):
                _sample(path, "text", 3, {"large", "ubuntu_irc"})
            with self.assertRaisesRegex(ValueError, "missing.*0"):
                _sample(path, "text", 3, {"large", "missing"}, allow_shortfall=True)


if __name__ == "__main__":
    unittest.main()
