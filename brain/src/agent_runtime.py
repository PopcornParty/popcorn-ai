"""Bounded orchestration core for trusted, host-provided development tools.

This module coordinates allowlisted handlers; it does not provide filesystem,
network, shell, GitHub or deployment access by itself. The host must supply
narrow handlers with their own sandboxing and resource limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping

from .constitution import action_requires_approval, APPROVAL_REQUIRED_ACTIONS, AUTONOMOUS_ACTIONS

MAX_STEPS = 20
MAX_PAYLOAD_BYTES = 16_384
BLOCKED_PAYLOAD_KEYS = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "private_key", "access_key", "authorization", "cookie", "credential",
    "bearer",
})

Handler = Callable[[dict[str, Any]], Any]
Approval = Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True)
class RuntimeStep:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    index: int
    action: str
    status: str
    detail: str


@dataclass(frozen=True)
class RuntimeReport:
    status: str
    records: tuple[StepRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "records": [
                {"index": r.index, "action": r.action, "status": r.status, "detail": r.detail}
                for r in self.records
            ],
        }


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Step payload must be an object")
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Step payload must contain JSON-safe values") from exc
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError("Step payload exceeds the size limit")

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Payload keys must be strings")
                normalized = key.casefold().replace("-", "_").replace(".", "_")
                key_parts = set(normalized.split("_"))
                if normalized in BLOCKED_PAYLOAD_KEYS or key_parts & {
                    "password", "passwd", "secret", "token", "authorization",
                    "cookie", "credential", "bearer",
                } or normalized.endswith("_api_key"):
                    raise ValueError("Payloads must not contain credentials or secret fields")
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload)
    return dict(payload)


def run_plan(
    steps: list[RuntimeStep] | tuple[RuntimeStep, ...],
    handlers: Mapping[str, Handler],
    *,
    approve: Approval | None = None,
    max_steps: int = MAX_STEPS,
) -> RuntimeReport:
    """Run a bounded plan using only host-registered handlers.

    Unknown actions fail closed. Protected actions need a positive approval
    callback for that exact action. Any failed or denied step stops the plan.
    """
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError("A plan must contain at least one step")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_STEPS}")
    if len(steps) > max_steps:
        raise ValueError("Plan exceeds the step limit")

    records: list[StepRecord] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, RuntimeStep):
            records.append(StepRecord(index, "<invalid>", "failed", "Invalid step object"))
            return RuntimeReport("failed", tuple(records))
        action = step.action
        if not isinstance(action, str) or not action or len(action) > 80:
            records.append(StepRecord(index, "<invalid>", "failed", "Invalid action"))
            return RuntimeReport("failed", tuple(records))
        if action not in AUTONOMOUS_ACTIONS and action not in APPROVAL_REQUIRED_ACTIONS:
            records.append(StepRecord(index, action, "blocked", "Unknown action; blocked by default"))
            return RuntimeReport("blocked", tuple(records))
        try:
            payload = _validate_payload(step.payload)
        except Exception as exc:
            records.append(StepRecord(index, action, "failed", type(exc).__name__))
            return RuntimeReport("failed", tuple(records))
        if action_requires_approval(action):
            try:
                granted = approve is not None and approve(action, payload) is True
            except Exception:
                granted = False
            if not granted:
                records.append(StepRecord(index, action, "approval_required", "Specific approval was not granted"))
                return RuntimeReport("approval_required", tuple(records))

        handler = handlers.get(action)
        if not callable(handler):
            records.append(StepRecord(index, action, "unavailable", "No trusted handler is registered"))
            return RuntimeReport("blocked", tuple(records))
        try:
            handler(payload)
        except Exception as exc:
            # Avoid copying exception text into the audit trail; it may contain
            # paths, snippets or other data supplied by a handler.
            records.append(StepRecord(index, action, "failed", type(exc).__name__))
            return RuntimeReport("failed", tuple(records))
        records.append(StepRecord(index, action, "completed", "Handler returned successfully"))

    return RuntimeReport("completed", tuple(records))
