import unittest

from src.planner import TaskPlan


class PlannerTests(unittest.TestCase):
    def test_parses_structured_candidate_read(self):
        plan = TaskPlan.from_dict({
            "goal": "Inspect source",
            "steps": [{
                "title": "Read file",
                "description": "Read a candidate source file",
                "tool": "read_repository_file",
                "arguments": {"path": "src/app.py"},
            }],
        })
        self.assertEqual(plan.steps[0].arguments, {"path": "src/app.py"})

    def test_candidate_patch_does_not_need_live_change_approval(self):
        plan = TaskPlan.from_dict({
            "goal": "Prepare isolated change",
            "steps": [{
                "title": "Write candidate",
                "description": "Write only inside isolated workspace",
                "tool": "propose_patch",
                "arguments": {"path": "src/app.py", "content": "VALUE = 1"},
            }],
        })
        self.assertFalse(plan.steps[0].requires_approval)

    def test_test_execution_still_requires_explicit_approval(self):
        with self.assertRaises(ValueError):
            TaskPlan.from_dict({
                "goal": "Run tests",
                "steps": [{
                    "title": "Run unit tests",
                    "description": "Execute a test suite",
                    "tool": "run_approved_test",
                    "arguments": {"suite": "unit"},
                }],
            })

    def test_round_trip_preserves_arguments(self):
        original = {
            "goal": "Inspect",
            "risk": "low",
            "steps": [{
                "title": "Read",
                "description": "Read a file",
                "tool": "read_repository_file",
                "arguments": {"path": "README.md"},
            }],
        }
        round_trip = TaskPlan.from_dict(TaskPlan.from_dict(original).to_dict())
        self.assertEqual(round_trip, TaskPlan.from_dict(original))

    def test_rejects_missing_or_invalid_arguments(self):
        base = {
            "goal": "Inspect",
            "steps": [{
                "title": "Read",
                "description": "Read a file",
                "tool": "read_repository_file",
                "arguments": {},
            }],
        }
        with self.assertRaises(ValueError):
            TaskPlan.from_dict(base)
        base["steps"][0]["arguments"] = {"path": 42}
        with self.assertRaises(ValueError):
            TaskPlan.from_dict(base)


if __name__ == "__main__":
    unittest.main()
