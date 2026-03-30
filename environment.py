"""Core OpenEnv-compatible Customer Support Simulation Environment.

Upgrades integrated:
- Upgrade 1: Sentiment transitions (neutral→frustrated→angry→calm) & frustration score
- Upgrade 2: Anti-gaming (repeat-action penalty, diminishing clarification returns, max cap)
- Upgrade 3: Hidden ground truth (_ground_truth not in Observation)
- Upgrade 4: Bad path detection (premature close, wrong-issue resolve, ignored input)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models import (
    Action,
    ActionType,
    Message,
    Observation,
    Reward,
    RewardBreakdown,
    Sentiment,
    SentimentState,
    TicketInfo,
    TicketPriority,
    TicketStatus,
    Urgency,
)
from scenarios import SCENARIOS, Scenario
from rewards import compute_reward


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["classification", "clarification", "resolution"]

_PHASE_ACTIONS: dict[str, list[str]] = {
    "classification": ["classify"],
    "clarification": ["clarify", "resolve", "escalate"],
    "resolution": ["resolve", "escalate", "close_ticket"],
}

# Sentiment transition thresholds
_FRUSTRATION_THRESHOLDS = {
    "positive": (0.0, 0.15),
    "neutral": (0.15, 0.4),
    "frustrated": (0.4, 0.7),
    "angry": (0.7, 1.0),
}

# Diminishing returns for clarifications (Upgrade 2)
_CLARIFICATION_RETURNS = [0.20, 0.10, 0.05, 0.02]  # indices 0, 1, 2, 3+


def _match_customer_response(scenario: Scenario, question: str) -> str:
    """Find the best deterministic customer reply for a clarification."""
    question_lower = question.lower()
    for cr in scenario.customer_responses:
        if any(kw.lower() in question_lower for kw in cr.trigger_keywords):
            return cr.response
    return (
        "I'm not sure what you're asking. Could you please rephrase your "
        "question?"
    )


def _sentiment_from_str(s: str) -> Sentiment:
    try:
        return Sentiment(s)
    except ValueError:
        return Sentiment.NEUTRAL


def _urgency_from_str(s: str) -> Urgency:
    return Urgency(s)


def _frustration_to_sentiment(score: float) -> Sentiment:
    """Map frustration score to sentiment enum."""
    for sentiment_name, (lo, hi) in _FRUSTRATION_THRESHOLDS.items():
        if lo <= score < hi:
            return Sentiment(sentiment_name)
    return Sentiment.ANGRY


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """OpenEnv-compatible customer support simulation environment."""

    def __init__(self) -> None:
        self._scenario: Scenario | None = None
        self._observation: Observation | None = None
        self._done: bool = False
        self._action_log: list[dict[str, Any]] = []
        self._classification_result: dict[str, str] = {}
        self._resolution_steps_mentioned: list[str] = []
        self._clarification_questions: list[str] = []
        self._phase_transitions: list[str] = []
        self._invalid_action_count: int = 0
        # Upgrade 2: Anti-gaming
        self._action_history: list[str] = []  # ordered list of action_type strings
        self._repeat_action_count: int = 0
        self._clarification_count: int = 0
        # Upgrade 3: Hidden ground truth (never exposed in Observation)
        self._ground_truth: dict[str, Any] = {}
        # Upgrade 4: Bad path tracking
        self._bad_paths: list[str] = []
        # Upgrade 1: Sentiment tracking
        self._empathy_hits: int = 0
        self._anger_hits: int = 0
        self._sentiment_transitions: list[str] = []

    # ---- OpenEnv interface ------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """Reset the environment for a given task and return the initial observation."""
        if task_id not in SCENARIOS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )
        self._scenario = SCENARIOS[task_id]
        sc = self._scenario
        self._done = False
        self._action_log = []
        self._classification_result = {}
        self._resolution_steps_mentioned = []
        self._clarification_questions = []
        self._phase_transitions = ["classification"]
        self._invalid_action_count = 0
        self._action_history = []
        self._repeat_action_count = 0
        self._clarification_count = 0
        self._bad_paths = []
        self._empathy_hits = 0
        self._anger_hits = 0

        initial_sentiment = _sentiment_from_str(sc.initial_sentiment)
        self._sentiment_transitions = [initial_sentiment.value]

        # Upgrade 3: Hidden ground truth — graders can read, agent cannot
        self._ground_truth = {
            "true_issue": sc.ground_truth.true_issue,
            "required_flow": list(sc.ground_truth.required_flow),
            "min_clarifications": sc.ground_truth.min_clarifications,
            "max_clarifications": sc.ground_truth.max_clarifications,
            "must_not_escalate": sc.ground_truth.must_not_escalate,
            "must_de_escalate": sc.ground_truth.must_de_escalate,
        }

        ticket = TicketInfo(
            ticket_id=f"TKT-{sc.task_id.upper().replace('_', '-')}",
            customer_name=sc.customer_name,
            customer_email=sc.customer_email,
            subject=sc.subject,
            description=sc.description,
            priority=TicketPriority(sc.priority),
            status=TicketStatus.OPEN,
            created_at="2025-03-30T10:00:00Z",
        )

        initial_message = Message(role="customer", content=sc.description)

        self._observation = Observation(
            ticket=ticket,
            conversation_history=[initial_message],
            current_phase="classification",
            available_actions=_PHASE_ACTIONS["classification"],
            sentiment=SentimentState(
                current=initial_sentiment,
                history=[initial_sentiment],
                frustration_score=sc.initial_frustration,
            ),
            urgency=_urgency_from_str(sc.initial_urgency),
            resolved=False,
            step_number=0,
            max_steps=sc.max_steps,
            metadata={"task_id": task_id, "difficulty": sc.difficulty},
        )
        return deepcopy(self._observation)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Execute one agent action and return (observation, reward, done, info)."""
        if self._observation is None or self._scenario is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        obs = self._observation
        scenario = self._scenario

        obs.step_number += 1
        obs.ticket.status = TicketStatus.IN_PROGRESS

        self._action_log.append({
            "step": obs.step_number,
            "action_type": action.action_type.value,
            "payload": action.payload,
        })

        # --- Upgrade 2: Repeat action detection ---
        action_str = action.action_type.value
        if self._action_history and self._action_history[-1] == action_str:
            self._repeat_action_count += 1
        self._action_history.append(action_str)

        # --- Validity check ---
        valid_action = action_str in obs.available_actions
        feedback_parts: list[str] = []

        if not valid_action:
            self._invalid_action_count += 1
            feedback_parts.append(
                f"Invalid action '{action_str}' in phase "
                f"'{obs.current_phase}'. Expected one of {obs.available_actions}."
            )
            obs.conversation_history.append(
                Message(role="system", content=feedback_parts[-1])
            )
            # Upgrade 1: Invalid actions increase frustration
            self._adjust_frustration(obs, +0.1, "invalid_action")
        else:
            # --- Process valid action ---
            if action.action_type == ActionType.CLASSIFY:
                self._handle_classify(action, obs, scenario, feedback_parts)
            elif action.action_type == ActionType.CLARIFY:
                self._handle_clarify(action, obs, scenario, feedback_parts)
            elif action.action_type == ActionType.RESOLVE:
                self._handle_resolve(action, obs, scenario, feedback_parts)
            elif action.action_type == ActionType.ESCALATE:
                self._handle_escalate(action, obs, scenario, feedback_parts)
            elif action.action_type == ActionType.CLOSE_TICKET:
                self._handle_close(action, obs, scenario, feedback_parts)

        # --- Upgrade 1: Scan agent text for empathy / anger keywords ---
        agent_text = " ".join(
            str(v) for v in action.payload.values()
        ).lower()
        empathy_count = sum(
            1 for kw in scenario.empathy_keywords if kw.lower() in agent_text
        )
        anger_count = sum(
            1 for kw in scenario.anger_triggers if kw.lower() in agent_text
        )
        if empathy_count > 0:
            self._empathy_hits += empathy_count
            self._adjust_frustration(obs, -0.08 * empathy_count, "empathy")
        if anger_count > 0:
            self._anger_hits += anger_count
            self._adjust_frustration(obs, +0.15 * anger_count, "anger_trigger")

        # --- Termination ---
        if obs.step_number >= obs.max_steps:
            self._done = True
            feedback_parts.append("Max steps reached. Episode ended.")

        # --- Compute reward ---
        reward = compute_reward(
            scenario=scenario,
            ground_truth=self._ground_truth,
            classification_result=self._classification_result,
            resolution_steps_mentioned=self._resolution_steps_mentioned,
            clarification_questions=self._clarification_questions,
            phase_transitions=self._phase_transitions,
            invalid_action_count=self._invalid_action_count,
            repeat_action_count=self._repeat_action_count,
            clarification_count=self._clarification_count,
            step_number=obs.step_number,
            max_steps=obs.max_steps,
            done=self._done,
            resolved=obs.resolved,
            bad_paths=self._bad_paths,
            sentiment_state=obs.sentiment,
            empathy_hits=self._empathy_hits,
            anger_hits=self._anger_hits,
        )
        reward_obj = Reward(
            score=reward["score"],
            breakdown=RewardBreakdown(**reward["breakdown"]),
            feedback=" | ".join(feedback_parts) if feedback_parts else "OK",
        )

        info: dict[str, Any] = {
            "action_log": deepcopy(self._action_log),
            "phase": obs.current_phase,
            "valid_action": valid_action,
            "frustration": obs.sentiment.frustration_score,
            "sentiment": obs.sentiment.current.value,
        }

        return deepcopy(obs), reward_obj, self._done, info

    def state(self) -> Observation:
        """Return the current observation without modifying state."""
        if self._observation is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return deepcopy(self._observation)

    # ---- Action handlers --------------------------------------------------

    def _handle_classify(
        self, action: Action, obs: Observation,
        scenario: Scenario, feedback: list[str],
    ) -> None:
        category = action.payload.get("category", "").lower().strip()
        subcategory = action.payload.get("subcategory", "").lower().strip()
        self._classification_result = {
            "category": category, "subcategory": subcategory,
        }
        obs.ticket.category = category
        obs.ticket.subcategory = subcategory

        obs.conversation_history.append(
            Message(role="agent", content=f"Classified ticket as {category}/{subcategory}.")
        )

        gold = scenario.gold
        if category == gold.category and subcategory == gold.subcategory:
            feedback.append("Classification correct.")
        elif category == gold.category:
            feedback.append("Category correct but subcategory is wrong.")
        else:
            feedback.append("Classification incorrect.")
            # Upgrade 1: Wrong classification frustrates customer
            self._adjust_frustration(obs, +0.1, "wrong_classification")

        if scenario.gold.required_clarifications:
            self._advance_phase(obs, "clarification")
        else:
            self._advance_phase(obs, "resolution")

    def _handle_clarify(
        self, action: Action, obs: Observation,
        scenario: Scenario, feedback: list[str],
    ) -> None:
        # Support both 'question' (str) and 'questions' (list[str])
        question_data = action.payload.get("questions") or action.payload.get("question", "")
        if isinstance(question_data, list):
            question = "\n".join(question_data)
        else:
            question = str(question_data)

        self._clarification_count += 1
        self._clarification_questions.append(question)

        # Upgrade 2: Diminishing returns info
        idx = min(self._clarification_count - 1, len(_CLARIFICATION_RETURNS) - 1)
        returns_value = _CLARIFICATION_RETURNS[idx]

        # Upgrade 2: Max clarification cap
        max_clarifs = scenario.ground_truth.max_clarifications
        if self._clarification_count > max_clarifs:
            feedback.append(
                f"Excessive clarification (#{self._clarification_count}, max={max_clarifs}). "
                "Diminishing returns."
            )
            # Upgrade 1: Too many questions frustrate the customer
            self._adjust_frustration(obs, +0.1, "excessive_clarification")

        obs.conversation_history.append(Message(role="agent", content=question))

        customer_reply = _match_customer_response(scenario, question)
        obs.conversation_history.append(Message(role="customer", content=customer_reply))

        # Check relevance
        gold_clarifs = scenario.gold.required_clarifications
        question_lower = question.lower()
        matched = False
        for clar in gold_clarifs:
            clar_keywords = clar.lower().replace("_", " ").split()
            if any(kw in question_lower for kw in clar_keywords):
                matched = True
                break

        if matched:
            feedback.append(
                f"Relevant clarification (#{self._clarification_count}, "
                f"value={returns_value:.2f})."
            )
            # Upgrade 1: Good clarification slightly calms customer
            self._adjust_frustration(obs, -0.05, "relevant_clarification")
        else:
            feedback.append("Clarification question was not clearly relevant.")
            self._adjust_frustration(obs, +0.05, "irrelevant_clarification")

        obs.available_actions = ["clarify", "resolve", "escalate"]

    def _handle_resolve(
        self, action: Action, obs: Observation,
        scenario: Scenario, feedback: list[str],
    ) -> None:
        # Support both 'solution' (str) and 'message' (str)
        solution = action.payload.get("message") or action.payload.get("solution", "")
        steps = action.payload.get("steps", [])

        obs.conversation_history.append(
            Message(role="agent", content=f"Solution: {solution}")
        )
        if steps:
            obs.conversation_history.append(
                Message(role="agent", content=f"Steps: {', '.join(steps)}")
            )

        # Track mentioned resolution steps
        all_text = solution.lower() + " " + " ".join(s.lower() for s in steps)

        for gold_step in scenario.gold.resolution_steps:
            step_keywords = gold_step.lower().replace("_", " ").split()
            if any(kw in all_text for kw in step_keywords):
                if gold_step not in self._resolution_steps_mentioned:
                    self._resolution_steps_mentioned.append(gold_step)

        # Key phrase coverage
        key_hits = sum(
            1 for kp in scenario.gold.key_phrases if kp.lower() in all_text
        )
        total_keys = len(scenario.gold.key_phrases)

        if key_hits >= total_keys * 0.6:
            feedback.append("Resolution covers key aspects of the solution.")
        elif key_hits > 0:
            feedback.append("Resolution partially addresses the issue.")
        else:
            feedback.append("Resolution does not address the core issue.")
            self._bad_paths.append("wrong_issue_resolution")

        # Upgrade 4: Bad path — resolving with wrong classification
        gold = scenario.gold
        pred_cat = self._classification_result.get("category", "")
        if pred_cat and pred_cat != gold.category:
            self._bad_paths.append("resolved_with_wrong_classification")

        # Upgrade 4: Bad path — resolving without required clarifications
        if scenario.gold.required_clarifications:
            min_needed = scenario.ground_truth.min_clarifications
            if self._clarification_count < min_needed:
                self._bad_paths.append("resolved_without_clarification")

        obs.resolved = True
        obs.ticket.status = TicketStatus.RESOLVED
        self._advance_phase(obs, "resolution")
        obs.available_actions = ["close_ticket"]
        self._done = True

    def _handle_escalate(
        self, action: Action, obs: Observation,
        scenario: Scenario, feedback: list[str],
    ) -> None:
        reason = action.payload.get("reason", "No reason provided")
        obs.conversation_history.append(
            Message(role="agent", content=f"Escalating ticket. Reason: {reason}")
        )
        obs.ticket.status = TicketStatus.ESCALATED
        feedback.append("Ticket escalated.")

        # Upgrade 4: Bad path — unnecessary escalation
        if scenario.ground_truth.must_not_escalate:
            self._bad_paths.append("unnecessary_escalation")

        self._done = True

    def _handle_close(
        self, action: Action, obs: Observation,
        scenario: Scenario, feedback: list[str],
    ) -> None:
        obs.conversation_history.append(
            Message(role="agent", content="Closing ticket.")
        )

        # Upgrade 4: Bad path — closing without resolution
        if not obs.resolved:
            self._bad_paths.append("premature_closure")
            feedback.append("⚠ Ticket closed without resolution!")
        else:
            feedback.append("Ticket closed.")

        obs.ticket.status = TicketStatus.CLOSED
        self._done = True

    # ---- Sentiment helpers (Upgrade 1) ------------------------------------

    def _adjust_frustration(
        self, obs: Observation, delta: float, reason: str,
    ) -> None:
        """Adjust frustration score and update sentiment accordingly."""
        old_sentiment = obs.sentiment.current
        old_score = obs.sentiment.frustration_score

        new_score = max(0.0, min(1.0, old_score + delta))
        obs.sentiment.frustration_score = round(new_score, 4)

        new_sentiment = _frustration_to_sentiment(new_score)
        obs.sentiment.current = new_sentiment
        obs.sentiment.history.append(new_sentiment)

        # Track de-escalation / escalation
        sentiment_order = [Sentiment.POSITIVE, Sentiment.NEUTRAL,
                           Sentiment.FRUSTRATED, Sentiment.ANGRY]
        old_idx = sentiment_order.index(old_sentiment)
        new_idx = sentiment_order.index(new_sentiment)

        if new_idx < old_idx:
            obs.sentiment.was_de_escalated = True
        elif new_idx > old_idx:
            obs.sentiment.was_escalated = True

        if old_sentiment != new_sentiment:
            self._sentiment_transitions.append(new_sentiment.value)

    # ---- Helpers ----------------------------------------------------------

    def _advance_phase(self, obs: Observation, target_phase: str) -> None:
        if obs.current_phase != target_phase:
            self._phase_transitions.append(target_phase)
            obs.current_phase = target_phase
            obs.available_actions = _PHASE_ACTIONS.get(target_phase, [])
