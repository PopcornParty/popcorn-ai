"""Connect structured plans to the restricted candidate workspace.

Reads, bounded knowledge search and candidate-code writes stay inside the
supplied isolated workspace. This adapter never runs generated code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .agent_runtime import RuntimeStep, run_plan
from .planner import TaskPlan
from .workspace import CandidateWorkspace

MAX_SEARCH_FILES = 100
MAX_SEARCH_BYTES = 2_000_000
MAX_SEARCH_RESULTS = 5
MAX_SNIPPETS_PER_FILE = 3
MAX_SNIPPET_CHARS = 320


@dataclass(frozen=True)
class WorkspacePlanResult:
    status: str
    report: dict[str, Any]
    outputs: tuple[dict[str, Any], ...]


def _search_workspace(workspace: CandidateWorkspace, query: str) -> list[dict[str, Any]]:
    """Search bounded text in candidate files; never access the host repository."""
    if not isinstance(query, str) or not query.strip() or len(query) > 120:
        raise ValueError("Search query must contain 1–120 characters")
    terms = tuple(dict.fromkeys(query.casefold().split()))
    if not terms:
        raise ValueError("Search query must contain searchable words")

    results: list[dict[str, Any]] = []
    bytes_scanned = 0
    files_scanned = 0

    for relative_path in workspace.list_files():
        if files_scanned >= MAX_SEARCH_FILES or bytes_scanned >= MAX_SEARCH_BYTES:
            break
        candidate_path = workspace.root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            size = candidate_path.stat().st_size
        except OSError:
            continue
        if size < 0 or size > MAX_SEARCH_BYTES - bytes_scanned:
            continue

        content = workspace.read_text(relative_path)
        files_scanned += 1
        bytes_scanned += size
        snippets = []
        for line_number, line in enumerate(content.splitlines(), 1):
            folded = line.casefold()
            if all(term in folded for term in terms):
                snippets.append({
                    "line": line_number,
                    "text": line[:MAX_SNIPPET_CHARS],
                })
                if len(snippets) >= MAX_SNIPPETS_PER_FILE:
                    break
        if snippets:
            results.append({"path": relative_path, "matches": snippets})
            if len(results) >= MAX_SEARCH_RESULTS:
                break

    return results


def execute_workspace_plan(plan: TaskPlan, workspace: CandidateWorkspace) -> WorkspacePlanResult:
    """Execute supported plan steps only inside the supplied candidate workspace.

    The complete plan is preflighted before any file is read or changed.
    Candidate writes do not affect the live repository. Generated code is
    never executed by this adapter.
    """
    if not isinstance(plan, TaskPlan):
        raise ValueError("A validated TaskPlan is required")
    supported = {
        "read_repository_file", "propose_patch", "search_knowledge",
        "summarise_result",
    }
    unsupported = [step.tool for step in plan.steps if step.tool not in supported]
    if unsupported:
        raise ValueError("Plan contains tools not available in the workspace adapter")

    required = {
        "read_repository_file": {"path"},
        "propose_patch": {"path", "content"},
        "search_knowledge": {"query"},
        "summarise_result": {"summary"},
    }
    for step in plan.steps:
        if set(step.arguments) != required[step.tool]:
            raise ValueError("Executable step is missing required arguments")

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
        elif step.tool == "search_knowledge":
            runtime_steps.append(RuntimeStep("read_project_files", {"query": args["query"]}))
        elif step.tool == "summarise_result":
            runtime_steps.append(RuntimeStep("prepare_change_summary", {"summary": args["summary"]}))

    def read_file(payload):
        content = workspace.read_text(payload["path"])
        outputs.append({"tool": "read_repository_file", "path": payload["path"], "content": content})

    def write_file(payload):
        workspace.write_text(payload["path"], payload["content"])
        outputs.append({"tool": "propose_patch", "path": payload["path"], "status": "written_to_candidate_workspace"})

    def search_files(payload):
        outputs.append({
            "tool": "search_knowledge",
            "query": payload["query"],
            "results": _search_workspace(workspace, payload["query"]),
        })

    def prepare_summary(payload):
        outputs.append({"tool": "summarise_result", "summary": payload["summary"]})

    report = run_plan(
        runtime_steps,
        {
            "read_project_files": lambda payload: (
                search_files(payload) if "query" in payload else read_file(payload)
            ),
            "write_candidate_code": write_file,
            "prepare_change_summary": prepare_summary,
        },
    )
    return WorkspacePlanResult(report.status, report.to_dict(), tuple(outputs))
