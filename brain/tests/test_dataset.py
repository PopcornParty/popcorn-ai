import json
import tempfile
import unittest
from pathlib import Path
from validate_dataset import inspect_dataset


class DatasetValidationTests(unittest.TestCase):
    def write(self, records):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "data.jsonl"
        path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
        return path

    def record(self, text, **metadata):
        details = {"source": "unit test", "license": "test-only", "permission": "unit test only", "reviewed": True}
        details.update(metadata)
        return {"text": text, "metadata": details}

    def test_reviewed_records_with_provenance_pass(self):
        report = inspect_dataset(self.write([self.record("alpha"), self.record("bravo"), self.record("charlie")]))
        self.assertTrue(report["ready_for_training"])

    def test_missing_provenance_blocks_training(self):
        report = inspect_dataset(self.write([{"text": "alpha"}, self.record("bravo"), self.record("charlie")]))
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(report["missing_metadata_line_numbers"], [1])

    def test_unreviewed_record_blocks_training(self):
        report = inspect_dataset(self.write([self.record("alpha", reviewed=False), self.record("bravo"), self.record("charlie")]))
        self.assertFalse(report["ready_for_training"])

    def test_normalised_duplicate_blocks_training(self):
        report = inspect_dataset(self.write([self.record("Alpha  beta"), self.record("alpha beta"), self.record("charlie")]))
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(report["duplicate_records"], 1)

    def test_secret_and_email_are_flagged(self):
        report = inspect_dataset(self.write([self.record("api_key = 'abcdefghijk12345'"), self.record("contact person@example.com"), self.record("safe text")]))
        self.assertFalse(report["ready_for_training"])
        self.assertEqual(report["possible_sensitive_content_line_numbers"], [1, 2])

    def test_oversized_record_blocks_training(self):
        report = inspect_dataset(self.write([self.record("x" * 100001), self.record("bravo"), self.record("charlie")]))
        self.assertFalse(report["ready_for_training"])


if __name__ == "__main__":
    unittest.main()
