import tempfile
import unittest
from pathlib import Path

from src.planner import TaskPlan
from src.workspace import CandidateWorkspace
from src.workspace_tools import execute_workspace_plan


class WorkspaceToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = CandidateWorkspace(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_reads_candidate_file_and_returns_content(self):
        self.workspace.write_text("src/app.py", "print('hello')\n")
        plan = TaskPlan.from_dict({
            "goal": "Inspect a source file",
            "steps": [{
                "title": "Read app",
                "description": "Inspect the candidate source",
                "tool": "read_repository_file",
                "arguments": {"path": "src/app.py"},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs[0]["content"], "print('hello')\n")

    def test_plan_writes_only_to_candidate_workspace(self):
        plan = TaskPlan.from_dict({
            "goal": "Add a small module",
            "steps": [{
                "title": "Create module",
                "description": "Write candidate code",
                "tool": "propose_patch",
                "arguments": {"path": "src/new.py", "content": "VALUE = 42\n"},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.workspace.read_text("src/new.py"), "VALUE = 42\n")
        self.assertEqual(result.outputs[0]["status"], "written_to_candidate_workspace")

    def test_unsupported_tools_are_rejected_before_any_write(self):
        plan = TaskPlan.from_dict({
            "goal": "Do multiple things",
            "steps": [
                {
                    "title": "Write file",
                    "description": "Candidate write",
                    "tool": "propose_patch",
                    "arguments": {"path": "src/new.py", "content": "x = 1"},
                },
                {
                    "title": "Run tests",
                    "description": "Not yet connected",
                    "tool": "run_approved_test",
                    "requires_approval": True,
                    "arguments": {"suite": "unit"},
                },
            ],
        })
        with self.assertRaises(ValueError):
            execute_workspace_plan(plan, self.workspace)
        self.assertEqual(self.workspace.list_files(), [])

    def test_workspace_rejects_unsafe_path_without_writing(self):
        plan = TaskPlan.from_dict({
            "goal": "Attempt unsafe path",
            "steps": [{
                "title": "Write outside",
                "description": "This must be blocked",
                "tool": "propose_patch",
                "arguments": {"path": "../outside.py", "content": "blocked"},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.workspace.list_files(), [])

    def test_planner_rejects_unknown_argument_keys(self):
        with self.assertRaises(ValueError):
            TaskPlan.from_dict({
                "goal": "Bad arguments",
                "steps": [{
                    "title": "Read",
                    "description": "Read a file",
                    "tool": "read_repository_file",
                    "arguments": {"path": "app.py", "token": "not-allowed"},
                }],
            })


    def test_plan_searches_only_candidate_workspace(self):
        self.workspace.write_text(
            "docs/guide.md",
            "Popcorn AI uses a candidate workspace.\nThe live site is separate.\n",
        )
        self.workspace.write_text("src/other.py", "def unrelated():\n    return 1\n")
        plan = TaskPlan.from_dict({
            "goal": "Find candidate workspace notes",
            "steps": [{
                "title": "Search notes",
                "description": "Search candidate files for workspace references",
                "tool": "search_knowledge",
                "arguments": {"query": "candidate workspace"},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs[0]["tool"], "search_knowledge")
        self.assertEqual(result.outputs[0]["results"][0]["path"], "docs/guide.md")
        self.assertEqual(result.outputs[0]["results"][0]["matches"][0]["line"], 1)

    def test_plan_returns_prepared_summary(self):
        plan = TaskPlan.from_dict({
            "goal": "Prepare a change summary",
            "steps": [{
                "title": "Summarise",
                "description": "Return the prepared summary",
                "tool": "summarise_result",
                "arguments": {"summary": "Added bounded workspace search."},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs[0], {
            "tool": "summarise_result",
            "summary": "Added bounded workspace search.",
        })

    def test_search_rejects_overlong_query(self):
        plan = TaskPlan.from_dict({
            "goal": "Search",
            "steps": [{
                "title": "Search",
                "description": "Search candidate files",
                "tool": "search_knowledge",
                "arguments": {"query": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
            }],
        })
        result = execute_workspace_plan(plan, self.workspace)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.outputs, ())

if __name__ == "__main__":
    unittest.main()
