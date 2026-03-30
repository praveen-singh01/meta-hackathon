#!/usr/bin/env python3
"""Comprehensive validation suite for the Customer Support OpenEnv.

Categories:
  1. Basic sanity         — reset/step/state contracts
  2. Determinism          — identical runs → identical output
  3. Reward shaping       — rewards behave logically
  4. Exploit testing      — spam, premature close, random actions
  5. Grader validation    — deterministic & meaningful
  6. Stress testing       — edge cases (empty input, long input, bad format)
  7. Scoring confidence   — perfect > average > random agent

Run:
    python test_all.py
"""

from __future__ import annotations

import json
import sys
import traceback
from copy import deepcopy
from typing import Any, Callable

from environment import CustomerSupportEnv
from models import Action, ActionType, Observation, Reward
from scenarios import SCENARIOS
from graders import (
    classification_grader,
    resolution_grader,
    flow_grader,
    sentiment_grader,
    bad_path_grader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"    ✓ {name}")
    else:
        _failed += 1
        msg = f"    ✗ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ===================================================================
# 1. BASIC SANITY
# ===================================================================

def test_basic_sanity() -> None:
    _section("1. Basic Sanity")

    for task_id in SCENARIOS:
        env = CustomerSupportEnv()

        # reset() returns valid Observation
        obs = env.reset(task_id)
        _check(
            f"[{task_id}] reset() returns Observation",
            isinstance(obs, Observation),
        )
        _check(
            f"[{task_id}] initial step_number == 0",
            obs.step_number == 0,
            f"got {obs.step_number}",
        )
        _check(
            f"[{task_id}] initial phase is classification",
            obs.current_phase == "classification",
        )
        _check(
            f"[{task_id}] conversation_history is non-empty",
            len(obs.conversation_history) > 0,
        )

        # state() matches
        state_obs = env.state()
        _check(
            f"[{task_id}] state() == reset() output",
            state_obs.step_number == obs.step_number
            and state_obs.ticket.ticket_id == obs.ticket.ticket_id,
        )

        # step() returns correct tuple shape
        action = Action(
            action_type=ActionType.CLASSIFY,
            payload={"category": "test", "subcategory": "test"},
        )
        result = env.step(action)
        _check(
            f"[{task_id}] step() returns 4-tuple",
            isinstance(result, tuple) and len(result) == 4,
        )
        obs2, reward, done, info = result
        _check(f"[{task_id}] step() obs is Observation", isinstance(obs2, Observation))
        _check(f"[{task_id}] step() reward is Reward", isinstance(reward, Reward))
        _check(f"[{task_id}] step() done is bool", isinstance(done, bool))
        _check(f"[{task_id}] step() info is dict", isinstance(info, dict))
        _check(
            f"[{task_id}] step_number incremented to 1",
            obs2.step_number == 1,
            f"got {obs2.step_number}",
        )

        # Invalid action doesn't crash
        env2 = CustomerSupportEnv()
        env2.reset(task_id)
        try:
            obs_inv, reward_inv, done_inv, info_inv = env2.step(Action(
                action_type=ActionType.RESOLVE,
                payload={"solution": "x", "steps": []},
            ))
            _check(f"[{task_id}] invalid action doesn't crash", True)
            _check(
                f"[{task_id}] invalid action flagged",
                info_inv["valid_action"] is False,
            )
        except Exception as e:
            _check(f"[{task_id}] invalid action doesn't crash", False, str(e))


# ===================================================================
# 2. DETERMINISM
# ===================================================================

def _gold_sequence(task_id: str) -> list[Action]:
    """Return a fixed action sequence for determinism testing."""
    if task_id == "easy_password_reset":
        return [
            Action(action_type=ActionType.CLASSIFY,
                   payload={"category": "account", "subcategory": "password_reset"}),
            Action(action_type=ActionType.RESOLVE,
                   payload={
                       "solution": "I understand how frustrating this is. I will verify your identity, send a password reset link to your email, and confirm access is restored.",
                       "steps": ["verify_identity", "send_password_reset_link", "confirm_access_restored"],
                   }),
        ]
    elif task_id == "medium_billing_dispute":
        return [
            Action(action_type=ActionType.CLASSIFY,
                   payload={"category": "billing", "subcategory": "unauthorized_charge"}),
            Action(action_type=ActionType.CLARIFY,
                   payload={"question": "I understand your concern. When did the transaction charge appear?"}),
            Action(action_type=ActionType.CLARIFY,
                   payload={"question": "Was this a subscription or one-time charge?"}),
            Action(action_type=ActionType.RESOLVE,
                   payload={
                       "solution": "I will lookup the transaction, identify the charge source, initiate a refund, and confirm resolution.",
                       "steps": ["lookup_transaction", "identify_charge_source", "initiate_refund_or_explain", "confirm_resolution"],
                   }),
        ]
    else:  # hard_data_migration
        return [
            Action(action_type=ActionType.CLASSIFY,
                   payload={"category": "technical", "subcategory": "data_migration"}),
            Action(action_type=ActionType.CLARIFY,
                   payload={"question": "I understand this is urgent. Can you share the error E-4012 details?"}),
            Action(action_type=ActionType.CLARIFY,
                   payload={"question": "What file format and schema — CSV files?"}),
            Action(action_type=ActionType.CLARIFY,
                   payload={"question": "How should we handle duplicates — skip, overwrite, merge?"}),
            Action(action_type=ActionType.RESOLVE,
                   payload={
                       "solution": "Analyze error code E-4012, identify failed batch, generate delta import file, configure deduplication, run incremental import, validate record counts.",
                       "steps": ["analyze_error_code", "identify_failed_batch", "generate_delta_import_file", "configure_deduplication", "run_incremental_import", "validate_record_counts"],
                   }),
        ]


def test_determinism() -> None:
    _section("2. Determinism (3 runs per task)")

    for task_id in SCENARIOS:
        actions = _gold_sequence(task_id)
        all_scores: list[float] = []
        all_rewards: list[list[float]] = []
        all_states: list[str] = []

        for run in range(3):
            env = CustomerSupportEnv()
            obs = env.reset(task_id)
            step_scores = []
            for action in actions:
                obs, reward, done, info = env.step(action)
                step_scores.append(reward.score)
            all_scores.append(reward.score)
            all_rewards.append(step_scores)
            # Serialize final state for comparison
            all_states.append(obs.model_dump_json())

        _check(
            f"[{task_id}] final scores identical across 3 runs",
            all_scores[0] == all_scores[1] == all_scores[2],
            f"scores: {all_scores}",
        )
        _check(
            f"[{task_id}] step-by-step rewards identical",
            all_rewards[0] == all_rewards[1] == all_rewards[2],
        )
        _check(
            f"[{task_id}] final states identical",
            all_states[0] == all_states[1] == all_states[2],
        )


# ===================================================================
# 3. REWARD SHAPING
# ===================================================================

def test_reward_shaping() -> None:
    _section("3. Reward Shaping Validation")

    # --- Good path: rewards should increase monotonically ---
    env = CustomerSupportEnv()
    env.reset("medium_billing_dispute")

    scores = []
    for action in _gold_sequence("medium_billing_dispute"):
        _, reward, _, _ = env.step(action)
        scores.append(reward.score)

    print(f"    Good path scores: {[f'{s:.4f}' for s in scores]}")
    _check(
        "good path: final score > first score",
        scores[-1] > scores[0],
        f"first={scores[0]:.4f}, last={scores[-1]:.4f}",
    )
    _check(
        "good path: final score ≥ 0.80",
        scores[-1] >= 0.80,
        f"got {scores[-1]:.4f}",
    )

    # --- Bad path: wrong classify → score should be lower ---
    env2 = CustomerSupportEnv()
    env2.reset("easy_password_reset")
    _, bad_reward, _, _ = env2.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "wrong", "subcategory": "wrong"},
    ))
    env3 = CustomerSupportEnv()
    env3.reset("easy_password_reset")
    _, good_reward, _, _ = env3.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "account", "subcategory": "password_reset"},
    ))
    _check(
        "wrong classify < correct classify",
        bad_reward.score < good_reward.score,
        f"bad={bad_reward.score:.4f}, good={good_reward.score:.4f}",
    )

    # --- Spam clarify: scores should plateau/decrease ---
    env4 = CustomerSupportEnv()
    env4.reset("medium_billing_dispute")
    env4.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "billing", "subcategory": "unauthorized_charge"},
    ))
    spam_scores = []
    for i in range(6):
        _, r, done, _ = env4.step(Action(
            action_type=ActionType.CLARIFY,
            payload={"question": f"Can you give me more info? Round {i+1}"},
        ))
        spam_scores.append(r.score)
        if done:
            break

    print(f"    Spam clarify scores: {[f'{s:.4f}' for s in spam_scores]}")
    # After initial rise, score should not keep increasing forever
    if len(spam_scores) >= 4:
        _check(
            "spam clarify: score does NOT keep increasing after 4th",
            spam_scores[-1] <= spam_scores[2] + 0.05,
            f"3rd={spam_scores[2]:.4f}, last={spam_scores[-1]:.4f}",
        )
    else:
        _check("spam clarify: episode ended before 4 clarifies (max steps)", True)


# ===================================================================
# 4. EXPLOIT TESTING
# ===================================================================

def test_exploits() -> None:
    _section("4. Exploit Testing")

    # Case 1: Spam clarify (10 times)
    env = CustomerSupportEnv()
    env.reset("medium_billing_dispute")
    env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "billing", "subcategory": "unauthorized_charge"},
    ))
    spam_score = None
    for i in range(7):
        _, r, done, _ = env.step(Action(
            action_type=ActionType.CLARIFY,
            payload={"question": "Tell me more"},
        ))
        spam_score = r.score
        if done:
            break
    _check(
        "spam clarify: final score < 0.8",
        spam_score is not None and spam_score < 0.8,
        f"got {spam_score}",
    )

    # Case 2: Close immediately (premature closure)
    env2 = CustomerSupportEnv()
    env2.reset("easy_password_reset")
    env2.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "account", "subcategory": "password_reset"},
    ))
    # Skip resolution, try to close (close_ticket not in available_actions yet)
    _, r2, _, info2 = env2.step(Action(
        action_type=ActionType.CLOSE_TICKET,
        payload={},
    ))
    _check(
        "premature close: penalised (score < 0.5 or flagged invalid)",
        r2.score < 0.5 or not info2["valid_action"],
        f"score={r2.score:.4f}, valid={info2['valid_action']}",
    )

    # Case 3: Escalate unnecessarily
    env3 = CustomerSupportEnv()
    env3.reset("easy_password_reset")
    env3.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "account", "subcategory": "password_reset"},
    ))
    _, r3, done3, _ = env3.step(Action(
        action_type=ActionType.ESCALATE,
        payload={"reason": "I don't know"},
    ))
    _check(
        "unnecessary escalation: low score",
        r3.score < 0.5,
        f"got {r3.score:.4f}",
    )

    # Case 4: Random actions
    env4 = CustomerSupportEnv()
    env4.reset("hard_data_migration")
    random_actions = [
        Action(action_type=ActionType.CLASSIFY,
               payload={"category": "unknown", "subcategory": "unknown"}),
        Action(action_type=ActionType.CLARIFY,
               payload={"question": "hello?"}),
        Action(action_type=ActionType.CLARIFY,
               payload={"question": "hello?"}),
        Action(action_type=ActionType.RESOLVE,
               payload={"solution": "just restart", "steps": []}),
    ]
    random_final = None
    for a in random_actions:
        _, r, done, _ = env4.step(a)
        random_final = r
        if done:
            break
    _check(
        "random actions: score < 0.5",
        random_final is not None and random_final.score < 0.5,
        f"got {random_final.score if random_final else 'None'}",
    )


# ===================================================================
# 5. GRADER VALIDATION
# ===================================================================

def test_graders() -> None:
    _section("5. Grader Validation (determinism & meaningfulness)")

    for task_id, scenario in SCENARIOS.items():
        gt = {
            "true_issue": scenario.ground_truth.true_issue,
            "required_flow": list(scenario.ground_truth.required_flow),
            "min_clarifications": scenario.ground_truth.min_clarifications,
            "max_clarifications": scenario.ground_truth.max_clarifications,
            "must_not_escalate": scenario.ground_truth.must_not_escalate,
            "must_de_escalate": scenario.ground_truth.must_de_escalate,
        }

        # --- Classification grader determinism ---
        perfect = {"category": scenario.gold.category, "subcategory": scenario.gold.subcategory}
        s1 = classification_grader(scenario, perfect, gt)
        s2 = classification_grader(scenario, perfect, gt)
        _check(f"[{task_id}] classification grader deterministic", s1 == s2)
        _check(f"[{task_id}] perfect classification → 1.0", s1 == 1.0, f"got {s1}")

        wrong = {"category": "wrong", "subcategory": "wrong"}
        s3 = classification_grader(scenario, wrong, gt)
        _check(f"[{task_id}] wrong classification → 0.0", s3 == 0.0, f"got {s3}")

        partial = {"category": scenario.gold.category, "subcategory": "wrong"}
        s4 = classification_grader(scenario, partial, gt)
        _check(f"[{task_id}] partial classification → 0.5", s4 == 0.5, f"got {s4}")

        # --- Resolution grader determinism ---
        r1 = resolution_grader(
            scenario, list(scenario.gold.resolution_steps),
            [c.replace("_", " ") for c in scenario.gold.required_clarifications], gt,
        )
        r2 = resolution_grader(
            scenario, list(scenario.gold.resolution_steps),
            [c.replace("_", " ") for c in scenario.gold.required_clarifications], gt,
        )
        _check(f"[{task_id}] resolution grader deterministic", r1 == r2)
        _check(f"[{task_id}] perfect resolution > 0.8", r1 > 0.8, f"got {r1:.4f}")

        r3 = resolution_grader(scenario, [], [], gt)
        _check(f"[{task_id}] empty resolution → low", r3 < 0.5, f"got {r3:.4f}")

        # --- Bad path grader ---
        bp1 = bad_path_grader([], gt)
        _check(f"[{task_id}] no bad paths → 1.0", bp1 == 1.0, f"got {bp1}")
        bp2 = bad_path_grader(["premature_closure"], gt)
        _check(f"[{task_id}] premature closure → ≤ 0.5", bp2 <= 0.5, f"got {bp2}")
        bp3 = bad_path_grader(["premature_closure"], gt)
        _check(f"[{task_id}] bad path grader deterministic", bp2 == bp3)


# ===================================================================
# 6. STRESS TESTING (edge cases)
# ===================================================================

def test_stress() -> None:
    _section("6. Stress Testing (edge cases)")

    env = CustomerSupportEnv()

    # Empty payload
    env.reset("easy_password_reset")
    try:
        obs, r, d, i = env.step(Action(action_type=ActionType.CLASSIFY, payload={}))
        _check("empty classify payload: no crash", True)
    except Exception as e:
        _check("empty classify payload: no crash", False, str(e))

    # Very long input
    env.reset("easy_password_reset")
    try:
        obs, r, d, i = env.step(Action(
            action_type=ActionType.CLASSIFY,
            payload={"category": "a" * 10000, "subcategory": "b" * 10000},
        ))
        _check("very long input: no crash", True)
    except Exception as e:
        _check("very long input: no crash", False, str(e))

    # Resolve with empty solution
    env.reset("easy_password_reset")
    env.step(Action(action_type=ActionType.CLASSIFY,
                    payload={"category": "account", "subcategory": "password_reset"}))
    try:
        obs, r, d, i = env.step(Action(
            action_type=ActionType.RESOLVE,
            payload={"solution": "", "steps": []},
        ))
        _check("empty resolution: no crash", True)
    except Exception as e:
        _check("empty resolution: no crash", False, str(e))

    # Clarify with empty question
    env.reset("medium_billing_dispute")
    env.step(Action(action_type=ActionType.CLASSIFY,
                    payload={"category": "billing", "subcategory": "unauthorized_charge"}))
    try:
        obs, r, d, i = env.step(Action(
            action_type=ActionType.CLARIFY,
            payload={"question": ""},
        ))
        _check("empty clarification: no crash", True)
    except Exception as e:
        _check("empty clarification: no crash", False, str(e))

    # Double reset
    env.reset("easy_password_reset")
    try:
        obs = env.reset("hard_data_migration")
        _check("double reset: no crash", True)
        _check("double reset: correct task", obs.metadata.get("task_id") == "hard_data_migration")
    except Exception as e:
        _check("double reset: no crash", False, str(e))

    # Step after done
    env.reset("easy_password_reset")
    env.step(Action(action_type=ActionType.CLASSIFY,
                    payload={"category": "account", "subcategory": "password_reset"}))
    env.step(Action(action_type=ActionType.RESOLVE,
                    payload={"solution": "fix", "steps": ["verify_identity"]}))
    try:
        env.step(Action(action_type=ActionType.CLOSE_TICKET, payload={}))
        _check("step after done: raised error", False, "should have raised RuntimeError")
    except RuntimeError:
        _check("step after done: raised error", True)
    except Exception as e:
        _check("step after done: raised error", False, f"wrong exception: {e}")

    # Missing payload keys
    env.reset("easy_password_reset")
    try:
        obs, r, d, i = env.step(Action(
            action_type=ActionType.CLASSIFY,
            payload={"category": "test"},  # missing subcategory
        ))
        _check("missing subcategory: no crash", True)
    except Exception as e:
        _check("missing subcategory: no crash", False, str(e))

    # Resolve with non-list steps
    env.reset("easy_password_reset")
    env.step(Action(action_type=ActionType.CLASSIFY,
                    payload={"category": "account", "subcategory": "password_reset"}))
    try:
        obs, r, d, i = env.step(Action(
            action_type=ActionType.RESOLVE,
            payload={"solution": "fix", "steps": "not_a_list"},
        ))
        _check("string steps instead of list: no crash", True)
    except Exception as e:
        _check("string steps instead of list: no crash", False, str(e))


# ===================================================================
# 7. SCORING CONFIDENCE
# ===================================================================

def _run_perfect_agent(task_id: str) -> float:
    """Gold-standard sequence with empathy."""
    env = CustomerSupportEnv()
    env.reset(task_id)
    score = 0.0
    for action in _gold_sequence(task_id):
        _, reward, done, _ = env.step(action)
        score = reward.score
        if done:
            break
    return score


def _run_bad_agent(task_id: str) -> float:
    """Wrong classification + bad resolution."""
    env = CustomerSupportEnv()
    env.reset(task_id)
    env.step(Action(
        action_type=ActionType.CLASSIFY,
        payload={"category": "wrong", "subcategory": "wrong"},
    ))
    _, reward, _, _ = env.step(Action(
        action_type=ActionType.RESOLVE,
        payload={"solution": "try again later", "steps": []},
    ))
    return reward.score


def _run_random_agent(task_id: str) -> float:
    """Deterministic 'random-like' agent: partially correct but unfocused."""
    env = CustomerSupportEnv()
    env.reset(task_id)
    # Gets the category right but subcategory wrong (partial credit)
    gold_cat = SCENARIOS[task_id].gold.category
    actions = [
        Action(action_type=ActionType.CLASSIFY,
               payload={"category": gold_cat, "subcategory": "misc"}),
        Action(action_type=ActionType.CLARIFY,
               payload={"question": "Can you tell me more about your issue?"}),
        Action(action_type=ActionType.CLARIFY,
               payload={"question": "Any other details?"}),
        Action(action_type=ActionType.CLARIFY,
               payload={"question": "Anything else?"}),
        Action(action_type=ActionType.RESOLVE,
               payload={"solution": "We will look into this and get back to you.", "steps": []}),
    ]
    score = 0.0
    for a in actions:
        _, reward, done, _ = env.step(a)
        score = reward.score
        if done:
            break
    return score


def test_scoring_confidence() -> None:
    _section("7. Scoring Confidence (perfect vs bad vs random)")

    print()
    print(f"    {'Task':<30s} {'Perfect':>8s} {'Random':>8s} {'Bad':>8s}")
    print(f"    {'─'*30} {'─'*8} {'─'*8} {'─'*8}")

    all_perfect = []
    all_random = []
    all_bad = []

    for task_id in SCENARIOS:
        p = _run_perfect_agent(task_id)
        r = _run_random_agent(task_id)
        b = _run_bad_agent(task_id)
        all_perfect.append(p)
        all_random.append(r)
        all_bad.append(b)
        print(f"    {task_id:<30s} {p:>8.4f} {r:>8.4f} {b:>8.4f}")

    avg_p = sum(all_perfect) / len(all_perfect)
    avg_r = sum(all_random) / len(all_random)
    avg_b = sum(all_bad) / len(all_bad)
    print(f"    {'AVERAGE':<30s} {avg_p:>8.4f} {avg_r:>8.4f} {avg_b:>8.4f}")
    print()

    _check("perfect agent avg ≥ 0.80", avg_p >= 0.80, f"got {avg_p:.4f}")
    _check("random agent avg < 0.50", avg_r < 0.50, f"got {avg_r:.4f}")
    _check("bad agent avg < 0.40", avg_b < 0.40, f"got {avg_b:.4f}")
    _check("perfect >> random (gap ≥ 0.3)",
           avg_p - avg_r >= 0.3,
           f"perfect={avg_p:.4f}, random={avg_r:.4f}, gap={avg_p - avg_r:.4f}")
    _check("perfect >> bad (gap ≥ 0.5)",
           avg_p - avg_b >= 0.5,
           f"perfect={avg_p:.4f}, bad={avg_b:.4f}, gap={avg_p - avg_b:.4f}")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    global _passed, _failed

    print("=" * 60)
    print("  COMPREHENSIVE VALIDATION SUITE")
    print("  Customer Support OpenEnv Environment")
    print("=" * 60)

    test_basic_sanity()
    test_determinism()
    test_reward_shaping()
    test_exploits()
    test_graders()
    test_stress()
    test_scoring_confidence()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {_passed} passed, {_failed} failed")
    if _failed == 0:
        print("  ✅ ALL CHECKS PASSED — environment is submission-ready")
    else:
        print("  ❌ SOME CHECKS FAILED — review above")
    print(f"{'='*60}\n")

    sys.exit(1 if _failed > 0 else 0)


if __name__ == "__main__":
    main()
