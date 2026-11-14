"""Connect structured plans to the restricted candidate workspace.

Only repository-file reads and candidate-code writes are supported here.
No shell, test execution, Git, network, secrets or deployment adapters exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_runtime import RuntimeStep, run_plan
from .planner import TaskPlan
from .workspace import CandidateWorkspace


@dataclass(frozen=True)
class WorkspacePlanResult:
    status: str
    report: dict[str, Any]
    outputs: tuple[dict[str, Any], ...]


def execute_workspace_plan(plan: TaskPlan, workspace: CandidateWorkspace) -> WorkspacePlanResult:
    """Execute supported plan steps only inside the supplied candidate workspace.

    Preflight rejects unsupported tools before any file is read or changed.
    Candidate writes are autonomous because they cannot affect the live repo.
    """
    if not isinstance(plan, TaskPlan):
        raise ValueError("A validated TaskPlan is required")
    supported = {"read_repository_file", "propose_patch"}
    unsupported = [step.tool for step in plan.steps if step.tool not in supported]
    if unsupported:
        raise ValueError("Plan contains tools not available in the workspace adapter")

    outputs: list[dict[str, Any]] = []
    runtime_steps = []
    for step in plan.steps:
        args = step.arguments
        if step.tool == "read_repository_file":
            runtime_steps.append(RuntimeStep("read_project_files", {"path": args["path"]}))
        elif step.tool == "propose_patch":
            runtime_steps.append(RuntimeStep(
                "write_candidate_code",
                {"path": args["path"], "content": args["content"]},
            ))
        else:
            raise ValueError("Every executable plan step must select a supported tool")

    def read_file(payload):
        content = workspace.read_text(payload["path"])
        outputs.append({"tool": "read_repository_file", "path": payload["path"], "content": content})

    def write_file(payload):
        workspace.write_text(payload["path"], payload["content"])
        outputs.append({"tool": "propose_patch", "path": payload["path"], "status": "written_to_candidate_workspace"})

    report = run_plan(
        runtime_steps,
        {
            "read_project_files": read_file,
            "write_candidate_code": write_file,
        },
    )
    return WorkspacePlanResult(report.status, report.to_dict(), tuple(outputs))
