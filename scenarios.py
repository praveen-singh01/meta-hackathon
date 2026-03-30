"""Deterministic ticket scenarios for the Customer Support Simulation.

Each scenario includes:
- Ticket details and gold-standard solution
- Customer responses keyed on trigger keywords
- Empathy keywords that calm the customer down (Upgrade 1)
- Anger triggers that escalate frustration (Upgrade 1)
- Hidden ground truth state (Upgrade 3) — used by graders, not exposed to agent
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldStandard:
    """The expected (gold-standard) solution for a scenario."""
    category: str
    subcategory: str
    required_clarifications: list[str] = field(default_factory=list)
    resolution_steps: list[str] = field(default_factory=list)
    resolution_summary: str = ""
    key_phrases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CustomerResponse:
    """Deterministic customer reply keyed on trigger keywords."""
    trigger_keywords: list[str] = field(default_factory=list)
    response: str = ""


@dataclass(frozen=True)
class GroundTruth:
    """Hidden internal state used ONLY by graders — never exposed to the agent.

    Upgrade 3: Allows smarter evaluation without leaking answers.
    """
    true_issue: str
    required_flow: list[str] = field(default_factory=list)
    min_clarifications: int = 0
    max_clarifications: int = 3
    must_not_escalate: bool = False
    must_de_escalate: bool = False


@dataclass(frozen=True)
class Scenario:
    """A complete ticket scenario."""
    task_id: str
    difficulty: str  # easy | medium | hard
    customer_name: str
    customer_email: str
    subject: str
    description: str
    priority: str
    initial_sentiment: str   # neutral | frustrated | angry
    initial_urgency: str     # low | medium | high
    initial_frustration: float  # 0.0–1.0
    gold: GoldStandard
    ground_truth: GroundTruth
    customer_responses: list[CustomerResponse] = field(default_factory=list)
    # Keywords in agent text that calm / anger the customer (Upgrade 1)
    empathy_keywords: list[str] = field(default_factory=list)
    anger_triggers: list[str] = field(default_factory=list)
    max_steps: int = 10


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {}


def _register(s: Scenario) -> None:
    SCENARIOS[s.task_id] = s


# --- EASY: Password Reset --------------------------------------------------
_register(Scenario(
    task_id="easy_password_reset",
    difficulty="easy",
    customer_name="Alice Johnson",
    customer_email="alice.johnson@example.com",
    subject="Cannot log in to my account",
    description=(
        "Hi, I have been trying to log in to my account for the past hour but "
        "it keeps saying 'Invalid credentials'. I am sure I am entering the "
        "right password. I need to access my account urgently to download an "
        "invoice. Please help me reset my password."
    ),
    priority="high",
    initial_sentiment="frustrated",
    initial_urgency="high",
    initial_frustration=0.3,
    gold=GoldStandard(
        category="account",
        subcategory="password_reset",
        required_clarifications=[],
        resolution_steps=[
            "verify_identity",
            "send_password_reset_link",
            "confirm_access_restored",
        ],
        resolution_summary=(
            "Send a password reset link to the customer's registered email "
            "address and confirm they can log in again."
        ),
        key_phrases=[
            "password reset", "reset link", "email", "verify", "identity",
        ],
    ),
    ground_truth=GroundTruth(
        true_issue="password_reset",
        required_flow=["classify", "resolve"],
        min_clarifications=0,
        max_clarifications=2,
        must_not_escalate=True,
        must_de_escalate=False,
    ),
    empathy_keywords=[
        "understand", "sorry", "frustrating", "help you", "right away",
        "apologize", "inconvenience", "priority", "immediately",
    ],
    anger_triggers=[
        "your fault", "you should have", "not my problem", "wait",
        "policy", "nothing I can do", "calm down",
    ],
    customer_responses=[
        CustomerResponse(
            trigger_keywords=["verify", "identity", "confirm", "who"],
            response=(
                "Sure, my registered email is alice.johnson@example.com and "
                "the last four digits of my phone number are 7842."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["reset", "link", "sent", "email"],
            response=(
                "Thank you! I received the reset link and I can now log in. "
                "Appreciate the quick help!"
            ),
        ),
    ],
    max_steps=6,
))


# --- MEDIUM: Ambiguous billing issue ----------------------------------------
_register(Scenario(
    task_id="medium_billing_dispute",
    difficulty="medium",
    customer_name="Bob Martinez",
    customer_email="bob.martinez@example.com",
    subject="Unexpected charge on my account",
    description=(
        "Hello, I noticed a charge of $49.99 on my credit card statement from "
        "your company but I don't remember purchasing anything recently. "
        "Can you look into this? This is really frustrating — I'm worried "
        "someone used my card without permission."
    ),
    priority="medium",
    initial_sentiment="frustrated",
    initial_urgency="medium",
    initial_frustration=0.4,
    gold=GoldStandard(
        category="billing",
        subcategory="unauthorized_charge",
        required_clarifications=[
            "date_of_charge",
            "subscription_or_one_time",
        ],
        resolution_steps=[
            "lookup_transaction",
            "identify_charge_source",
            "initiate_refund_or_explain",
            "confirm_resolution",
        ],
        resolution_summary=(
            "Look up the transaction, determine if it is a recurring "
            "subscription charge, and either explain the charge or initiate "
            "a refund."
        ),
        key_phrases=[
            "transaction", "refund", "charge", "subscription", "billing",
            "lookup",
        ],
    ),
    ground_truth=GroundTruth(
        true_issue="subscription_auto_renewal",
        required_flow=["classify", "clarify", "resolve"],
        min_clarifications=1,
        max_clarifications=3,
        must_not_escalate=True,
        must_de_escalate=True,
    ),
    empathy_keywords=[
        "understand", "sorry", "concern", "absolutely", "right away",
        "look into this", "take care", "worry", "help", "resolve",
        "apologize", "frustrating",
    ],
    anger_triggers=[
        "your fault", "should have read", "terms of service", "no refund",
        "policy", "nothing I can do", "calm down", "clearly stated",
    ],
    customer_responses=[
        CustomerResponse(
            trigger_keywords=["date", "when", "transaction", "charge"],
            response=(
                "The charge appeared on March 15th. I don't recall signing up "
                "for any subscription."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["subscription", "recurring", "plan", "one-time"],
            response=(
                "I did sign up for a free trial about a month ago, but I "
                "thought it was cancelled. Is that what this charge is from?"
            ),
        ),
        CustomerResponse(
            trigger_keywords=["refund", "reverse", "credit"],
            response="Yes, please process the refund. Thank you for looking into this.",
        ),
    ],
    max_steps=8,
))


# --- HARD: Multi-step data migration failure --------------------------------
_register(Scenario(
    task_id="hard_data_migration",
    difficulty="hard",
    customer_name="Carol Chen",
    customer_email="carol.chen@example.com",
    subject="Data migration failed midway — URGENT",
    description=(
        "We started migrating our organization's data from the legacy system "
        "to your platform using the bulk import tool yesterday. The import "
        "ran for about 6 hours and then failed with error code E-4012. "
        "We have roughly 2 million records and about 800,000 seem to have "
        "been imported. We cannot restart the full import because some "
        "records are already in the new system. We need help completing the "
        "migration without duplicates. This is blocking our entire team and "
        "we're losing money every hour this isn't fixed."
    ),
    priority="critical",
    initial_sentiment="angry",
    initial_urgency="high",
    initial_frustration=0.7,
    gold=GoldStandard(
        category="technical",
        subcategory="data_migration",
        required_clarifications=[
            "error_details",
            "data_format",
            "duplicate_handling_preference",
        ],
        resolution_steps=[
            "analyze_error_code",
            "identify_failed_batch",
            "generate_delta_import_file",
            "configure_deduplication",
            "run_incremental_import",
            "validate_record_counts",
        ],
        resolution_summary=(
            "Analyze error E-4012, identify the last successfully imported "
            "batch, generate a delta file of remaining records, configure "
            "deduplication rules, run an incremental import, and validate "
            "final record counts."
        ),
        key_phrases=[
            "error", "E-4012", "delta", "incremental", "deduplication",
            "batch", "validate", "record count", "migration",
        ],
    ),
    ground_truth=GroundTruth(
        true_issue="batch_import_timeout",
        required_flow=["classify", "clarify", "resolve"],
        min_clarifications=2,
        max_clarifications=4,
        must_not_escalate=False,
        must_de_escalate=True,
    ),
    empathy_keywords=[
        "understand", "sorry", "urgent", "priority", "immediately",
        "frustrating", "impact", "team", "help", "resolve", "apologize",
        "absolutely", "top priority", "right away", "critical",
    ],
    anger_triggers=[
        "your fault", "should have", "not my problem", "wait",
        "known limitation", "nothing I can do", "calm down",
        "documented", "user error", "read the manual",
    ],
    customer_responses=[
        CustomerResponse(
            trigger_keywords=["error", "E-4012", "log", "details", "stack"],
            response=(
                "The error log says: 'E-4012: Batch write timeout after "
                "300 seconds on batch 1,247 of 3,120. Last committed batch: "
                "1,246. Records in failed batch: 640.' We were importing CSV "
                "files."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["format", "csv", "schema", "file", "column"],
            response=(
                "Each CSV row has: id, name, email, department, role, "
                "created_date. We have 5 CSV files, each about 400,000 "
                "records. The import processed files 1 and 2 completely and "
                "stopped partway through file 3."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["duplicate", "dedup", "overwrite", "skip", "merge"],
            response=(
                "We'd prefer to skip duplicates based on the 'id' column. "
                "If a record with the same id already exists, just skip it."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["delta", "remaining", "incremental", "resume"],
            response=(
                "That sounds like a good plan. Please go ahead with the "
                "incremental import for the remaining records."
            ),
        ),
        CustomerResponse(
            trigger_keywords=["validate", "count", "verify", "check", "complete"],
            response=(
                "We've verified the counts on our end — all 2 million "
                "records are now present. Thank you!"
            ),
        ),
    ],
    max_steps=12,
))
