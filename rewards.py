"""Dense reward computation for the Customer Support Simulation.

Reward weights:
    +0.15  correct classification
    +0.15  relevant clarification (with diminishing returns)
    +0.30  correct resolution
    +0.15  flow / closure quality
    +0.15  sentiment management (de-escalation)
    +0.10  no bad paths (clean agent behaviour)

Penalties (Upgrade 2 — anti-gaming):
    - Repeat same action consecutively → −0.1 per repeat
    - Diminishing clarification returns (see _CLARIFICATION_RETURNS)
    - Excessive clarifications penalised

Penalties (Upgrade 4 — bad path):
    - Premature closure → up to −0.5
    - Wrong-issue resolution → −0.3
    - Resolved without clarification → −0.2

Final score is clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any

from graders import (
    classification_grader,
    resolution_grader,
    flow_grader,
    sentiment_grader,
    bad_path_grader,
)
from models import SentimentState
from scenarios import Scenario

# Diminishing returns schedule for clarifications (Upgrade 2)
_CLARIFICATION_RETURNS = [0.20, 0.10, 0.05, 0.02]


def compute_reward(
    *,
    scenario: Scenario,
    ground_truth: dict[str, Any],
    classification_result: dict[str, str],
    resolution_steps_mentioned: list[str],
    clarification_questions: list[str],
    phase_transitions: list[str],
    invalid_action_count: int,
    repeat_action_count: int,
    clarification_count: int,
    step_number: int,
    max_steps: int,
    done: bool,
    resolved: bool,
    bad_paths: list[str],
    sentiment_state: SentimentState,
    empathy_hits: int,
    anger_hits: int,
) -> dict[str, Any]:
    """Return a dict with 'score' (float) and 'breakdown' (dict).

    Dense rewards: the score reflects partial progress at every step.
    """

    # --- Sub-scores via graders ---
    cls_raw = classification_grader(scenario, classification_result, ground_truth)
    res_raw = resolution_grader(
        scenario, resolution_steps_mentioned, clarification_questions, ground_truth,
    )
    flw_raw = flow_grader(
        scenario, phase_transitions, invalid_action_count,
        step_number, max_steps, ground_truth,
    )
    sent_raw = sentiment_grader(scenario, sentiment_state, ground_truth)
    bp_raw = bad_path_grader(bad_paths, ground_truth)

    # --- Weighted components ---
    classification_score = 0.15 * cls_raw
    resolution_score = 0.30 * res_raw
    flow_score = 0.15 * flw_raw
    sentiment_score = 0.15 * sent_raw
    bad_path_component = 0.10 * bp_raw

    # --- Clarification with diminishing returns (Upgrade 2) ---
    total_clarifs = len(scenario.gold.required_clarifications)
    if total_clarifs > 0:
        # Calculate effective clarification value
        clarif_value = 0.0
        for i in range(clarification_count):
            idx = min(i, len(_CLARIFICATION_RETURNS) - 1)
            clarif_value += _CLARIFICATION_RETURNS[idx]

        # Proportional to required clarifications matched
        matched = 0
        for req in scenario.gold.required_clarifications:
            req_kws = req.lower().replace("_", " ").split()
            for q in clarification_questions:
                if any(kw in q.lower() for kw in req_kws):
                    matched += 1
                    break
        match_ratio = matched / total_clarifs

        # Cap clarification score at weight of 0.15
        clarification_score = min(0.15, 0.15 * match_ratio)
    else:
        clarification_score = 0.15  # no clarifications needed → full marks

    # --- Penalties ---
    step_penalty = 0.0
    invalid_penalty = 0.0
    repeat_penalty = 0.0
    bad_path_penalty_val = 0.0

    # Invalid actions
    if invalid_action_count > 0:
        invalid_penalty = min(0.3, invalid_action_count * 0.1)

    # Upgrade 2: Repeat action penalty
    if repeat_action_count > 0:
        repeat_penalty = min(0.3, repeat_action_count * 0.1)

    # Premature closure (done but not resolved)
    if done and not resolved:
        step_penalty += 0.5

    # Step efficiency
    if max_steps > 0 and step_number / max_steps > 0.8:
        step_penalty += 0.05

    # Upgrade 4: bad path penalties (already factored via bad_path_component,
    # but add explicit penalty for severe cases)
    severe_bad = sum(1 for bp in bad_paths if bp in (
        "premature_closure", "wrong_issue_resolution",
    ))
    if severe_bad > 0:
        bad_path_penalty_val = min(0.3, severe_bad * 0.15)

    # --- Composite ---
    raw = (
        classification_score
        + clarification_score
        + resolution_score
        + flow_score
        + sentiment_score
        + bad_path_component
        - step_penalty
        - invalid_penalty
        - repeat_penalty
        - bad_path_penalty_val
    )
    score = max(0.0, min(1.0, raw))

    breakdown = {
        "classification_score": round(classification_score, 4),
        "clarification_score": round(clarification_score, 4),
        "resolution_score": round(resolution_score, 4),
        "closure_score": round(flow_score, 4),
        "flow_score": round(flw_raw, 4),
        "sentiment_score": round(sentiment_score, 4),
        "step_penalty": round(step_penalty, 4),
        "invalid_action_penalty": round(invalid_penalty, 4),
        "repeat_action_penalty": round(repeat_penalty, 4),
        "bad_path_penalty": round(bad_path_penalty_val, 4),
    }
    return {"score": round(score, 4), "breakdown": breakdown}
