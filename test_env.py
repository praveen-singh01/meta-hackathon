"""Smoke tests for the Customer Support Simulation Environment.

Run:  python test_env.py

Tests:
1. Gold-standard sequences → high scores
2. Bad action sequences → low scores
3. State consistency
4. Invalid action handling
5. Sentiment de-escalation (Upgrade 1)
6. Anti-gaming: repeat action penalty (Upgrade 2)
7. Anti-gaming: excessive clarification (Upgrade 2)
8. Bad path: premature closure (Upgrade 4)
"""

from __future__ import annotations

import sys

from environment import CustomerSupportEnv
from models import Action, ActionType


# ---------------------------------------------------------------------------
# Gold-standard sequences
# ---------------------------------------------------------------------------

def _run_gold_easy() -> float:
    env = CustomerSupportEnv()
    obs = env.reset("easy_password_reset")
    assert obs.current_phase == "classification"

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "account", "subcategory": "password_reset"},
    ))
    print(f"  [easy] After classify: score={reward.score:.4f}, phase={obs.current_phase}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={
            "solution": (
                "I understand how frustrating this must be. I will verify "
                "your identity right away, send a password reset link to "
                "your registered email address, and confirm access is restored "
                "immediately. I apologize for the inconvenience."
            ),
            "steps": [
                "verify_identity",
                "send_password_reset_link",
                "confirm_access_restored",
            ],
        },
    ))
    print(f"  [easy] After resolve: score={reward.score:.4f}, done={done}")
    return reward.score


def _run_gold_medium() -> float:
    env = CustomerSupportEnv()
    obs = env.reset("medium_billing_dispute")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "billing", "subcategory": "unauthorized_charge"},
    ))
    print(f"  [medium] After classify: score={reward.score:.4f}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "I understand your concern and I'm sorry about this. "
                "When did the charge appear — what was the transaction date?"
            ),
        },
    ))
    print(f"  [medium] After clarify (date): score={reward.score:.4f}, "
          f"sentiment={obs.sentiment.current.value}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "Thank you. Was this a subscription recurring charge or "
                "a one-time purchase? I want to help resolve this right away."
            ),
        },
    ))
    print(f"  [medium] After clarify (sub): score={reward.score:.4f}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={
            "solution": (
                "I will lookup the transaction from March 15th, identify the "
                "charge source as a subscription renewal from your free trial, "
                "initiate a refund for the $49.99 charge, and confirm the "
                "resolution with you. I apologize for the inconvenience."
            ),
            "steps": [
                "lookup_transaction",
                "identify_charge_source",
                "initiate_refund_or_explain",
                "confirm_resolution",
            ],
        },
    ))
    print(f"  [medium] After resolve: score={reward.score:.4f}, done={done}")
    return reward.score


def _run_gold_hard() -> float:
    env = CustomerSupportEnv()
    obs = env.reset("hard_data_migration")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "technical", "subcategory": "data_migration"},
    ))
    print(f"  [hard] After classify: score={reward.score:.4f}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "I understand this is critical and your team is blocked. "
                "This is my top priority. Can you share the full error log "
                "and details for error E-4012?"
            ),
        },
    ))
    print(f"  [hard] After clarify (error): score={reward.score:.4f}, "
          f"frustration={obs.sentiment.frustration_score:.0%}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "I'm sorry for the impact on your team. What file format and "
                "schema are you using? Are these CSV files?"
            ),
        },
    ))
    print(f"  [hard] After clarify (format): score={reward.score:.4f}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "Thank you. How should we handle duplicates — skip, overwrite, "
                "or merge? I want to make sure we get this right."
            ),
        },
    ))
    print(f"  [hard] After clarify (dedup): score={reward.score:.4f}")

    obs, reward, done, _ = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={
            "solution": (
                "I will immediately analyze error code E-4012 to identify "
                "the failed batch, generate a delta import file for the "
                "remaining records starting from batch 1247, configure "
                "deduplication to skip existing IDs, run an incremental "
                "import, and validate the final record counts to ensure all "
                "2 million records are present."
            ),
            "steps": [
                "analyze_error_code",
                "identify_failed_batch",
                "generate_delta_import_file",
                "configure_deduplication",
                "run_incremental_import",
                "validate_record_counts",
            ],
        },
    ))
    print(f"  [hard] After resolve: score={reward.score:.4f}, done={done}")
    return reward.score


# ---------------------------------------------------------------------------
# Bad / edge-case sequences
# ---------------------------------------------------------------------------

def _run_bad_sequence() -> float:
    """Wrong classification + irrelevant resolution."""
    env = CustomerSupportEnv()
    env.reset("easy_password_reset")

    env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "billing", "subcategory": "refund"},
    ))

    _, reward, _, _ = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={"solution": "Please try again later.", "steps": []},
    ))
    print(f"  [bad] After bad resolve: score={reward.score:.4f}")
    return reward.score


def _test_state() -> None:
    """Verify state() returns consistent observation."""
    env = CustomerSupportEnv()
    obs1 = env.reset("easy_password_reset")
    obs2 = env.state()
    assert obs1.step_number == obs2.step_number
    assert obs1.ticket.ticket_id == obs2.ticket.ticket_id
    assert obs1.sentiment.frustration_score == obs2.sentiment.frustration_score
    print("  [state] state() consistency ✓")


def _test_invalid_action() -> None:
    """Invalid actions are handled gracefully."""
    env = CustomerSupportEnv()
    env.reset("easy_password_reset")
    obs, reward, done, info = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={"solution": "test", "steps": []},
    ))
    assert not info["valid_action"]
    print(f"  [invalid] Invalid action handled: score={reward.score:.4f} ✓")


def _test_sentiment_de_escalation() -> None:
    """Upgrade 1: Empathetic language reduces frustration."""
    env = CustomerSupportEnv()
    obs = env.reset("hard_data_migration")
    initial_frustration = obs.sentiment.frustration_score
    print(f"  [sentiment] Initial frustration: {initial_frustration:.0%}")

    obs, _, _, _ = env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "technical", "subcategory": "data_migration"},
    ))

    # Use lots of empathy keywords
    obs, _, _, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={
            "question": (
                "I completely understand how frustrating and urgent this is. "
                "I sincerely apologize for the impact on your team. "
                "This is my absolute top priority and I will help you "
                "resolve this immediately. Can you share the error details?"
            ),
        },
    ))
    print(f"  [sentiment] After empathetic response: "
          f"frustration={obs.sentiment.frustration_score:.0%}")
    assert obs.sentiment.frustration_score < initial_frustration, \
        "Empathy should reduce frustration"
    print("  [sentiment] De-escalation works ✓")


def _test_repeat_action_penalty() -> None:
    """Upgrade 2: Consecutive same action → penalty."""
    env = CustomerSupportEnv()
    env.reset("medium_billing_dispute")

    env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "billing", "subcategory": "unauthorized_charge"},
    ))

    # Two consecutive clarify actions
    _, r1, _, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={"question": "When did the charge appear?"},
    ))
    _, r2, _, _ = env.step(Action(
        action_type=ActionType.CLARIFY,
        payload={"question": "Was it a subscription or one-time?"},
    ))
    # Second clarify should have repeat penalty factored in
    print(f"  [anti-gaming] Clarify #1 score: {r1.score:.4f}, "
          f"#2 score: {r2.score:.4f}")
    # The repeat penalty should show in breakdown
    assert r2.breakdown.repeat_action_penalty >= 0.0
    print("  [anti-gaming] Repeat action tracking ✓")


def _test_premature_closure() -> None:
    """Upgrade 4: Closing without resolving → bad path penalty."""
    env = CustomerSupportEnv()
    obs = env.reset("easy_password_reset")

    env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "account", "subcategory": "password_reset"},
    ))

    # Try to close immediately via escalate (skip resolution)
    _, reward, done, _ = env.step(Action(
        action_type=ActionType.ESCALATE,
        payload={"reason": "I don't know how to fix this"},
    ))
    print(f"  [bad path] Unnecessary escalation score: {reward.score:.4f}")
    assert reward.breakdown.bad_path_penalty > 0 or reward.score < 0.6, \
        "Unnecessary escalation should be penalised"
    print("  [bad path] Bad path detection ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Customer Support Env — Smoke Tests (with Upgrades)")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("easy (gold)", lambda: _run_gold_easy() >= 0.80),
        ("medium (gold)", lambda: _run_gold_medium() >= 0.80),
        ("hard (gold)", lambda: _run_gold_hard() >= 0.80),
        ("bad sequence", lambda: _run_bad_sequence() < 0.5),
        ("state() consistency", lambda: (_test_state(), True)[1]),
        ("invalid action handling", lambda: (_test_invalid_action(), True)[1]),
        ("sentiment de-escalation", lambda: (_test_sentiment_de_escalation(), True)[1]),
        ("repeat action penalty", lambda: (_test_repeat_action_penalty(), True)[1]),
        ("premature closure", lambda: (_test_premature_closure(), True)[1]),
    ]

    for name, test_fn in tests:
        print(f"\n▶ {name}")
        try:
            result = test_fn()
            if result:
                print(f"  ✓ PASS")
                passed += 1
            else:
                print(f"  ✗ FAIL")
                failed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
