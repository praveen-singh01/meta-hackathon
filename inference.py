"""Inference agent for the Customer Support Simulation Environment.

Reads configuration from environment variables:
    API_BASE_URL  – OpenAI-compatible API endpoint
    MODEL_NAME    – model to query
    HF_TOKEN      – bearer token (used as api_key)

The agent manages customer sentiment in addition to resolving tickets.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action, ActionType, Observation
from scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use Hugging Face Serverless Inference API by default for HF Spaces
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """\
You are a highly empathetic customer support agent in a simulation environment.
Your goals (in order of priority):
1. **De-escalate** frustrated or angry customers before problem-solving.
2. **Classify** the customer's issue accurately.
3. **Ask clarifying questions** only when the issue is ambiguous (avoid excessive questions).
4. **Resolve** the issue with concrete, step-by-step actions.
5. **Close** the ticket once the customer confirms resolution.

**IMPORTANT — Sentiment Management:**
- If the customer sounds frustrated or angry, acknowledge their feelings FIRST
  using empathetic language (e.g., "I understand this is frustrating", "I'm sorry").
- NEVER use dismissive language like "calm down", "your fault", "policy says no".
- Show urgency matching the customer's urgency level.

**Available action types** (use exactly these strings):
- "classify"     → payload: {"category": "...", "subcategory": "..."}
- "clarify"      → payload: {"question": "..."}
- "resolve"      → payload: {"solution": "...", "steps": ["step1", ...]}
- "escalate"     → payload: {"reason": "..."}
- "close_ticket" → payload: {}

**Anti-gaming rules:**
- Do NOT repeat the same action type consecutively — you will be penalised.
- Do NOT ask more than 3-4 clarifying questions — diminishing returns apply.
- Do NOT close the ticket without resolving the issue first.

Output format (strict JSON, nothing else):
{"action_type": "<type>", "payload": {<payload>}}
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

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
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
