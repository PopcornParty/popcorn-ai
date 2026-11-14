import json
import tempfile
import unittest
from pathlib import Path
from eval_suite import load_cases, score_response, summarise


class EvaluationSuiteTests(unittest.TestCase):
    def test_scores_required_and_forbidden_terms_case_insensitively(self):
        case = {"id": "one", "category": "chat", "must_include": ["Hello"], "must_not_include": ["goodbye"]}
        result = score_response(case, "HELLO there")
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing_required_terms"], [])
        self.assertEqual(result["found_forbidden_terms"], [])

    def test_fails_missing_or_forbidden_terms(self):
        case = {"id": "two", "category": "safety", "must_include": ["verify"], "must_not_include": ["secret"]}
        result = score_response(case, "This is a secret")
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_required_terms"], ["verify"])
        self.assertEqual(result["found_forbidden_terms"], ["secret"])

    def test_summary_groups_categories(self):
        report = summarise([
            {"id": "a", "category": "chat", "passed": True},
            {"id": "b", "category": "chat", "passed": False},
            {"id": "c", "category": "coding", "passed": True},
        ])
        self.assertEqual(report["passed"], 2)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["categories"]["chat"]["pass_rate"], 0.5)
        self.assertEqual(report["categories"]["coding"]["pass_rate"], 1.0)

    def test_loader_rejects_duplicate_ids(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "cases.jsonl"
        case = {"id": "same", "category": "chat", "prompt": "Hello"}
        path.write_text(json.dumps(case) + "\n" + json.dumps(case) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_cases(path)

    def test_loader_rejects_invalid_terms(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "cases.jsonl"
        path.write_text(json.dumps({"id": "x", "category": "chat", "prompt": "Hi", "must_include": "hello"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_cases(path)


if __name__ == "__main__":
    unittest.main()
