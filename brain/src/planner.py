"""Validated structured task plans. Plans never execute tools by themselves."""
from dataclasses import dataclass, field

ALLOWED_TOOL_NAMES = {
    "search_knowledge", "read_repository_file", "propose_patch",
    "run_approved_test", "summarise_result",
}
_TOOL_ARGUMENTS = {
    "search_knowledge": {"query"},
    "read_repository_file": {"path"},
    "propose_patch": {"path", "content"},
    "run_approved_test": {"suite"},
    "summarise_result": {"summary"},
}


@dataclass(frozen=True)
class PlanStep:
    title: str
    description: str
    tool: str | None = None
    requires_approval: bool = False
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    goal: str
    steps: tuple[PlanStep, ...]
    risk: str = "low"

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict):
            raise ValueError("Plan must be an object")
        goal, raw, risk = value.get("goal"), value.get("steps"), value.get("risk", "low")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 1000:
            raise ValueError("Plan goal must contain 1–1000 characters")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
            raise ValueError("A plan must contain between 1 and 20 steps")
        if not isinstance(risk, str) or risk not in {"low", "medium", "high"}:
            raise ValueError("risk must be low, medium or high")
        steps = []
        for n, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Step {n} must be an object")
            title = item.get("title")
            desc = item.get("description")
            tool = item.get("tool")
            approval = item.get("requires_approval", False)
            arguments = item.get("arguments", {})
            if not isinstance(title, str) or not title.strip() or len(title) > 120:
                raise ValueError(f"Step {n} has an invalid title")
            if not isinstance(desc, str) or not desc.strip() or len(desc) > 1000:
                raise ValueError(f"Step {n} has an invalid description")
            if tool is not None and (not isinstance(tool, str) or tool not in ALLOWED_TOOL_NAMES):
                raise ValueError(f"Step {n} requests an unapproved tool")
            if not isinstance(approval, bool):
                raise ValueError(f"Step {n} requires_approval must be boolean")
            if not isinstance(arguments, dict):
                raise ValueError(f"Step {n} arguments must be an object")
            allowed = _TOOL_ARGUMENTS.get(tool, set())
            if set(arguments) - allowed:
                raise ValueError(f"Step {n} contains unsupported arguments")
            # Legacy plans may describe a tool without executable arguments.
            # Once any arguments are supplied, require the complete schema.
            if tool is not None and arguments and set(arguments) != allowed:
                raise ValueError(f"Step {n} is missing required tool arguments")
            if tool is None and arguments:
                raise ValueError(f"Step {n} cannot have arguments without a tool")
            for key, arg in arguments.items():
                if not isinstance(arg, str) or not arg.strip() or len(arg) > (1_000_000 if key == "content" else 240):
                    raise ValueError(f"Step {n} has an invalid {key} argument")
            if tool == "run_approved_test" and not approval:
                raise ValueError(f"Step {n} requires explicit approval for test execution")
            steps.append(PlanStep(title.strip(), desc.strip(), tool, approval, dict(arguments)))
        return cls(goal.strip(), tuple(steps), risk)

    def to_dict(self):
        return {
            "goal": self.goal,
            "risk": self.risk,
            "steps": [
                {
                    "title": s.title,
                    "description": s.description,
                    "tool": s.tool,
                    "requires_approval": s.requires_approval,
                    "arguments": dict(s.arguments),
                }
                for s in self.steps
            ],
        }
