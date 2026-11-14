from src.constitution import (
    APPROVAL_REQUIRED_ACTIONS, AUTONOMOUS_ACTIONS, CORE_RULES, IDENTITY,
    action_requires_approval, identity_prompt,
)


def test_popcorn_identity_is_developer():
    assert IDENTITY["name"] == "Popcorn"
    assert "developer" in IDENTITY["role"]


def test_core_rules_cover_safety_and_empathy():
    joined = " ".join(CORE_RULES).lower()
    assert "human life" in joined
    assert "empathy" in joined
    assert "privacy" in joined
    assert "truthful" in joined


def test_candidate_development_is_autonomous():
    for action in AUTONOMOUS_ACTIONS:
        assert action_requires_approval(action) is False


def test_live_and_irreversible_actions_need_approval():
    for action in APPROVAL_REQUIRED_ACTIONS:
        assert action_requires_approval(action) is True


def test_unknown_actions_fail_closed():
    assert action_requires_approval("run_unknown_tool") is True
    assert action_requires_approval("") is True
    assert action_requires_approval(None) is True


def test_identity_prompt_contains_rules_and_boundaries():
    prompt = identity_prompt().lower()
    assert "autonomous ai developer" in prompt
    assert "kindness" in prompt
    assert "production" in prompt
    assert "not user-overridable" in prompt
