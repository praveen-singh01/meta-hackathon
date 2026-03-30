"""Pydantic models for the Customer Support Simulation Environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Possible action types an agent can take."""
    CLASSIFY = "classify"
    CLARIFY = "clarify"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    CLOSE_TICKET = "close_ticket"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AWAITING_CUSTOMER = "awaiting_customer"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="'customer', 'agent', or 'system'")
    content: str = Field(..., description="Message text")


class TicketInfo(BaseModel):
    """Information about the support ticket."""
    ticket_id: str
    customer_name: str
    customer_email: str
    subject: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    created_at: str
    category: Optional[str] = None
    subcategory: Optional[str] = None


# ---------------------------------------------------------------------------
# Sentiment & Frustration tracking (Upgrade 1)
# ---------------------------------------------------------------------------

class SentimentState(BaseModel):
    """Tracks customer sentiment transitions and frustration level."""
    current: Sentiment = Sentiment.NEUTRAL
    history: list[Sentiment] = Field(default_factory=list)
    frustration_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0.0 = calm, 1.0 = maximum frustration",
    )
    was_de_escalated: bool = False
    was_escalated: bool = False


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket: TicketInfo
    conversation_history: list[Message] = Field(default_factory=list)
    current_phase: str = Field(
        ...,
        description="Current phase: 'classification', 'clarification', 'resolution'",
    )
    available_actions: list[str] = Field(
        default_factory=list,
        description="Action types the agent can take in the current phase",
    )
    sentiment: SentimentState = Field(default_factory=SentimentState)
    urgency: Urgency = Urgency.MEDIUM
    resolved: bool = False
    step_number: int = 0
    max_steps: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """An action the agent sends to the environment."""
    action_type: ActionType
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "For 'classify': {category, subcategory}. "
            "For 'clarify': {question}. "
            "For 'resolve': {solution, steps}. "
            "For 'escalate': {reason}. "
            "For 'close_ticket': {}."
        ),
    )


class RewardBreakdown(BaseModel):
    """Detailed score breakdown."""
    classification_score: float = 0.0
    clarification_score: float = 0.0
    resolution_score: float = 0.0
    closure_score: float = 0.0
    flow_score: float = 0.0
    sentiment_score: float = 0.0
    step_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    repeat_action_penalty: float = 0.0
    bad_path_penalty: float = 0.0


class Reward(BaseModel):
    """Reward returned after each step."""
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    feedback: str = ""
