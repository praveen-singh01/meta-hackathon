"""Deterministic graders for the Customer Support Simulation.

Each grader returns a float score in [0.0, 1.0].

Upgrades:
- Upgrade 3: Graders now accept ground_truth for smarter evaluation
- Upgrade 4: bad_path_grader detects shortcuts and penalises them
"""

from __future__ import annotations

from typing import Any

from models import SentimentState
from scenarios import Scenario


# ---------------------------------------------------------------------------
# 1. Classification grader
# ---------------------------------------------------------------------------

def classification_grader(
    scenario: Scenario,
    classification_result: dict[str, str],
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Score the agent's classification (0.0 – 1.0).

    - Exact match on category AND subcategory → 1.0
    - Category correct, subcategory wrong   → 0.5
    - Both wrong                             → 0.0
    """
    gold = scenario.gold
    predicted_cat = classification_result.get("category", "").lower().strip()
    predicted_sub = classification_result.get("subcategory", "").lower().strip()

    if not predicted_cat:
        return 0.0

    # --- Fuzzy/Keyword matching for category ---
    gold_cat = gold.category.lower()
    cat_match = (predicted_cat == gold_cat) or (gold_cat in predicted_cat) or (predicted_cat in gold_cat)

    # --- Fuzzy/Keyword matching for subcategory ---
    gold_sub = gold.subcategory.lower()
    sub_match = (predicted_sub == gold_sub) or (gold_sub in predicted_sub) or (predicted_sub in gold_sub)
    
    # Also allow matching across words (e.g. "password reset" matches "password_reset")
    if not sub_match:
        gold_sub_clean = gold_sub.replace("_", " ")
        sub_match = (gold_sub_clean in predicted_sub) or (predicted_sub in gold_sub_clean)

    if cat_match and sub_match:
        return 1.0
    if cat_match:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# 2. Resolution grader
# ---------------------------------------------------------------------------

def resolution_grader(
    scenario: Scenario,
    resolution_steps_mentioned: list[str],
    clarification_questions: list[str],
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Score how well the agent resolved the issue (0.0 – 1.0).

    Uses ground truth min_clarifications for smarter scoring (Upgrade 3).
    """
    gold = scenario.gold
    gt = ground_truth or {}

    # --- Resolution step coverage ---
    total_gold_steps = len(gold.resolution_steps)
    if total_gold_steps > 0:
        matched_steps = sum(
            1 for gs in gold.resolution_steps
            if gs in resolution_steps_mentioned
        )
        step_score = matched_steps / total_gold_steps
    else:
        step_score = 1.0

    # --- Clarification coverage ---
    total_clarifs = len(gold.required_clarifications)
    if total_clarifs > 0:
        matched_clarifs = 0
        for req in gold.required_clarifications:
            req_keywords = req.lower().replace("_", " ").split()
            for q in clarification_questions:
                if any(kw in q.lower() for kw in req_keywords):
                    matched_clarifs += 1
                    break
        clarif_score = matched_clarifs / total_clarifs
    else:
        clarif_score = 1.0

    # --- Ground truth check: did agent meet minimum clarifications? ---
    min_clarifs = gt.get("min_clarifications", 0)
    if min_clarifs > 0 and len(clarification_questions) < min_clarifs:
        clarif_penalty = 0.2  # didn't ask enough questions
    else:
        clarif_penalty = 0.0

    if total_clarifs > 0:
        return max(0.0, 0.5 * step_score + 0.5 * clarif_score - clarif_penalty)
    return max(0.0, step_score - clarif_penalty)


# ---------------------------------------------------------------------------
# 3. Flow grader
# ---------------------------------------------------------------------------

def flow_grader(
    scenario: Scenario,
    phase_transitions: list[str],
    invalid_action_count: int,
    step_number: int,
    max_steps: int,
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Score the agent's adherence to proper workflow (0.0 – 1.0).

    Uses ground truth required_flow for smarter evaluation (Upgrade 3).
    """
    score = 1.0
    gt = ground_truth or {}

    # --- Phase order check (use ground truth if available) ---
    expected_flow = gt.get("required_flow", None)
    if expected_flow:
        # Map action types to phases
        flow_to_phase = {
            "classify": "classification",
            "clarify": "clarification",
            "resolve": "resolution",
        }
        expected_phases = [flow_to_phase.get(f, f) for f in expected_flow]
    else:
        expected_phases = ["classification"]
        if scenario.gold.required_clarifications:
            expected_phases.append("clarification")
        expected_phases.append("resolution")

    exp_idx = 0
    for phase in phase_transitions:
        while exp_idx < len(expected_phases) and expected_phases[exp_idx] != phase:
            exp_idx += 1
        if exp_idx >= len(expected_phases):
            score -= 0.3
            break
        exp_idx += 1

    # --- Invalid action penalty ---
    if invalid_action_count > 0:
        score -= min(0.3, invalid_action_count * 0.1)

    # --- Step efficiency ---
    if max_steps > 0:
        usage_ratio = step_number / max_steps
        if usage_ratio > 0.8:
            score -= 0.1

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 4. Sentiment grader (Upgrade 1)
# ---------------------------------------------------------------------------

def sentiment_grader(
    scenario: Scenario,
    sentiment_state: SentimentState,
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Score how well the agent managed customer sentiment (0.0 – 1.0).

    Rewards de-escalation, penalises escalation of frustration.
    """
    score = 0.5  # baseline

    gt = ground_truth or {}

    # Reward: frustration decreased from initial
    initial_frustration = scenario.initial_frustration
    final_frustration = sentiment_state.frustration_score
    delta = initial_frustration - final_frustration  # positive = improved

    if delta > 0:
        score += min(0.4, delta)  # up to +0.4 for calming user
    elif delta < 0:
        score -= min(0.4, abs(delta))  # up to -0.4 for angering user

    # Bonus: if de-escalation was required and achieved
    if gt.get("must_de_escalate") and sentiment_state.was_de_escalated:
        score += 0.1

    # Penalty: if de-escalation was required but not achieved
    if gt.get("must_de_escalate") and not sentiment_state.was_de_escalated:
        score -= 0.1

    # Penalty: sentiment was escalated (got worse)
    if sentiment_state.was_escalated:
        score -= 0.1

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 5. Bad path grader (Upgrade 4)
# ---------------------------------------------------------------------------

def bad_path_grader(
    bad_paths: list[str],
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """Detect and penalise shortcut / bad behaviours (0.0 – 1.0).

    1.0 = no bad paths detected, 0.0 = severe violations.
    """
    score = 1.0

    penalties = {
        "premature_closure": 0.5,
        "wrong_issue_resolution": 0.3,
        "resolved_with_wrong_classification": 0.25,
        "resolved_without_clarification": 0.2,
        "unnecessary_escalation": 0.2,
    }

    for bp in bad_paths:
        score -= penalties.get(bp, 0.1)

    return max(0.0, min(1.0, score))
