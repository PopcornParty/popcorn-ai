import tempfile
import unittest
from pathlib import Path

from src.workspace import CandidateWorkspace, WorkspaceError


class CandidateWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = CandidateWorkspace(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_reads_and_writes_candidate_code(self):
        self.workspace.write_text("src/example.py", "print('candidate')\n")
        self.assertEqual(self.workspace.read_text("src/example.py"), "print('candidate')\n")
        self.assertEqual(self.workspace.list_files(), ["src/example.py"])

    def test_rejects_path_traversal_and_absolute_paths(self):
        for path in ("../outside.py", "/tmp/outside.py", "src/../../outside.py"):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.workspace.write_text(path, "blocked")

    def test_rejects_hidden_secret_and_dependency_paths(self):
        for path in (".env", ".git/config.json", ".ssh/id_rsa.py", "node_modules/pkg/index.js", "secrets/key.py"):
            with self.subTest(path=path), self.assertRaises(WorkspaceError):
                self.workspace.write_text(path, "blocked")

    def test_rejects_unapproved_file_types_and_oversized_content(self):
        with self.assertRaises(WorkspaceError):
            self.workspace.write_text("payload.exe", "blocked")
        with self.assertRaises(WorkspaceError):
            self.workspace.write_text("large.py", "x" * 1_000_001)

    def test_rejects_symlink_escape(self):
        outside = self.root.parent / (self.root.name + "-outside.py")
        outside.write_text("outside", encoding="utf-8")
        try:
            (self.root / "escape.py").symlink_to(outside)
            with self.assertRaises(WorkspaceError):
                self.workspace.read_text("escape.py")
        finally:
            outside.unlink(missing_ok=True)

    def test_rejects_non_utf8_candidate_file(self):
        (self.root / "bad.py").write_bytes(b"\xff")
        with self.assertRaises(WorkspaceError):
            self.workspace.read_text("bad.py")

    def test_root_must_be_an_existing_directory(self):
        with self.assertRaises(WorkspaceError):
            CandidateWorkspace(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
