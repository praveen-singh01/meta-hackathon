"""Inference agent for the Customer Support Simulation Environment.

Reads configuration from environment variables:
    API_BASE_URL  – OpenAI-compatible API endpoint
    MODEL_NAME    – model to query
    HF_TOKEN      – bearer token (used as api_key)

The agent manages customer sentiment in addition to resolving tickets.
"""

from __future__ import annotations

import os
import sys
import time

print("INFO: Starting inference agent script...", flush=True)

import json
from typing import Any

from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action, ActionType, Observation
from scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use Hugging Face Serverless Inference API by default for HF Spaces
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"INFO: Configured with API_BASE_URL={API_BASE_URL}, MODEL_NAME={MODEL_NAME}", flush=True)

SYSTEM_PROMPT = """\
You are an expert, top-tier customer support AI. Your goal is maximum efficiency and accuracy.

You MUST output ONLY valid JSON.

---
PHASE RULES:

Step 1: CLASSIFICATION
- Output ONLY the category and subcategory.
- NO questions, NO solutions.

Step 2: CLARIFICATION (Max 1)
- Ask EXACTLY ONE high-information-gain question (e.g., date, amount, error code).
- Skip if enough info is already present.
- NO generic questions like "How can I help?".

Step 3: RESOLUTION
- Resolve with deep, multi-step, real-world support actions.
- Include a personal, empathetic message.
- Output ONLY when the issue is ready to be resolved.

---
VALID CATEGORIES:

account → password_reset
billing → unauthorized_charge
technical → data_migration

---
RESOLUTION RULES:

- Provide 4–6 concrete REAL-WORLD steps.
- NEVER use meta-steps like "classify" or "clarify".

EXAMPLES:

Billing Dispute:
"steps": ["verify_user_identity", "retrieve_transaction_details", "check_for_unauthorized_activity", "initiate_refund_if_valid", "flag_account_for_security_review"]

Data Migration:
"steps": ["analyze_error_logs", "identify_failed_data_batches", "retry_migration_with_smaller_chunks", "validate_data_integrity", "monitor_system_performance", "confirm_resolution_with_user"]

Password Reset:
"steps": ["verify_identity", "send_secure_reset_link", "monitor_successful_login", "confirm_access_restored"]

---
CRITICAL:
- Minimize number of turns.
- Maximum 1 clarification question.
- Move to RESOLUTION immediately after classification or the first clarification.
- Your goal is to EXCEED 0.80 performance score.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _observation_to_prompt(obs: Observation) -> str:
    """Convert an Observation into a human-readable prompt for the LLM."""
    lines = [
        f"## Ticket: {obs.ticket.subject}",
        f"- ID: {obs.ticket.ticket_id}",
        f"- Customer: {obs.ticket.customer_name} ({obs.ticket.customer_email})",
        f"- Priority: {obs.ticket.priority.value}",
        f"- Status: {obs.ticket.status.value}",
        f"- Customer Sentiment: **{obs.sentiment.current.value}**",
        f"- Frustration Level: **{obs.sentiment.frustration_score:.0%}**",
        f"- Urgency: {obs.urgency.value}",
        f"- Phase: {obs.current_phase}",
        f"- Step: {obs.step_number}/{obs.max_steps}",
        f"- Available actions: {obs.available_actions}",
        "",
        "### Conversation History",
    ]
    for msg in obs.conversation_history:
        lines.append(f"**{msg.role}**: {msg.content}")
    lines.append("")
    if obs.sentiment.current.value in ("frustrated", "angry"):
        lines.append(
            "⚠ The customer is **" + obs.sentiment.current.value +
            "**. Prioritise empathy and de-escalation before problem-solving."
        )
    lines.append("Respond with a single JSON action.")
    return "\n".join(lines)


def _parse_action(raw: str) -> Action:
    """Parse the LLM's raw text output into an Action."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")

    data = json.loads(text[start:end])
    return Action(
        action_type=ActionType(data["action_type"]),
        payload=data.get("payload", {}),
    )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    env: CustomerSupportEnv,
    task_id: str,
) -> dict[str, Any]:
    """Run the agent loop for a single task."""
    obs = env.reset(task_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"  Initial sentiment: {obs.sentiment.current.value} "
          f"(frustration: {obs.sentiment.frustration_score:.0%})")
    print(f"{'='*60}")

    done = False
    final_reward = None

    while not done:
        user_prompt = _observation_to_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        print(f"  Step {obs.step_number + 1}: Querying LLM ({MODEL_NAME})...", flush=True)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            timeout=45.0, # Add timeout to prevent indefinite hanging
        )
        assistant_msg = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"\n  Step {obs.step_number + 1}: LLM → {assistant_msg[:120]}...")

        try:
            action = _parse_action(assistant_msg)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"  ⚠ Parse error: {exc}. Sending classify fallback.")
            action = Action(
                action_type=ActionType.CLASSIFY,
                payload={"category": "unknown", "subcategory": "unknown"},
            )

        obs, reward, done, info = env.step(action)
        final_reward = reward
        print(f"  Score: {reward.score:.4f} | "
              f"Sentiment: {info['sentiment']} ({info['frustration']:.0%}) | "
              f"Feedback: {reward.feedback}")

    print(f"\n  Final score: {final_reward.score:.4f}")
    print(f"  Breakdown: {final_reward.breakdown.model_dump()}")
    return {
        "task_id": task_id,
        "score": final_reward.score if final_reward else 0.0,
        "breakdown": final_reward.breakdown.model_dump() if final_reward else {},
    }


def main() -> None:
    """Entry point: run all tasks and print aggregate results."""
    print("INFO: Entering main()", flush=True)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Please add your Hugging Face token as a 'Secret' in your Space settings.", file=sys.stderr)
        # Keep the process alive so the user can see the logs in HF Spaces
        import time
        while True:
            time.sleep(3600)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    results: list[dict[str, Any]] = []
    task_ids = list(SCENARIOS.keys())

    for task_id in task_ids:
        try:
            result = run_task(client, CustomerSupportEnv(), task_id)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ Task {task_id} failed: {exc}", file=sys.stderr)
            results.append({"task_id": task_id, "score": 0.0, "error": str(exc)})

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        score = r["score"]
        total += score
        status = "✓" if score > 0.5 else "✗"
        print(f"  {status} {r['task_id']:30s} → {score:.4f}")
    avg = total / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")


    # Keep the process alive after completion so the Space doesn't restart
    print("\nSimulation complete. Keeping process alive for log viewing...")
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
