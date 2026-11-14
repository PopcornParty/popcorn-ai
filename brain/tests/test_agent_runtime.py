import unittest

from src.agent_runtime import RuntimeStep, run_plan


class AgentRuntimeTests(unittest.TestCase):
    def test_runs_registered_autonomous_steps_in_order(self):
        seen = []
        handlers = {
            "analyse_task": lambda payload: seen.append(("analyse", payload["goal"])),
            "run_candidate_tests": lambda payload: seen.append(("test", payload["suite"])),
        }
        report = run_plan(
            [
                RuntimeStep("analyse_task", {"goal": "fix bug"}),
                RuntimeStep("run_candidate_tests", {"suite": "unit"}),
            ],
            handlers,
        )
        self.assertEqual(report.status, "completed")
        self.assertEqual([item[0] for item in seen], ["analyse", "test"])
        self.assertEqual([r.status for r in report.records], ["completed", "completed"])

    def test_unknown_action_fails_closed(self):
        report = run_plan([RuntimeStep("run_shell_command", {})], {})
        self.assertEqual(report.status, "blocked")
        self.assertEqual(report.records[0].status, "blocked")

    def test_protected_action_needs_specific_approval(self):
        calls = []
        handlers = {"deploy_production": lambda payload: calls.append(payload)}
        denied = run_plan([RuntimeStep("deploy_production", {"target": "live"})], handlers)
        self.assertEqual(denied.status, "approval_required")
        self.assertEqual(calls, [])
        allowed = run_plan(
            [RuntimeStep("deploy_production", {"target": "live"})],
            handlers,
            approve=lambda action, payload: action == "deploy_production",
        )
        self.assertEqual(allowed.status, "completed")
        self.assertEqual(len(calls), 1)

    def test_stops_after_handler_failure(self):
        seen = []
        def fail(_payload):
            raise RuntimeError("sensitive details should not enter audit")
        handlers = {
            "analyse_task": fail,
            "plan_task": lambda payload: seen.append(payload),
        }
        report = run_plan(
            [RuntimeStep("analyse_task"), RuntimeStep("plan_task")],
            handlers,
        )
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.records[-1].detail, "RuntimeError")
        self.assertEqual(seen, [])

    def test_rejects_secret_fields(self):
        report = run_plan(
            [RuntimeStep("read_project_files", {"api_key": "do-not-log"})],
            {"read_project_files": lambda payload: self.fail("handler must not run")},
        )
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.records[0].detail, "ValueError")

    def test_rejects_oversized_plans_and_payloads(self):
        with self.assertRaises(ValueError):
            run_plan([RuntimeStep("plan_task") for _ in range(21)], {})
        with self.assertRaises(ValueError):
            run_plan(
                [RuntimeStep("plan_task", {"text": "x" * 20_000})],
                {"plan_task": lambda payload: None},
            )

    def test_report_is_json_serializable_and_does_not_include_payload(self):
        report = run_plan(
            [RuntimeStep("plan_task", {"goal": "private project name"})],
            {"plan_task": lambda payload: None},
        )
        rendered = report.to_dict()
        self.assertEqual(rendered["status"], "completed")
        self.assertNotIn("private project name", str(rendered))


if __name__ == "__main__":
    unittest.main()
