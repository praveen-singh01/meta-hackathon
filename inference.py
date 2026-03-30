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
import json
import re
from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn

from openai import OpenAI

from environment import CustomerSupportEnv
from models import Action, ActionType, Observation, Reward
from scenarios import SCENARIOS

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

print("INFO: Starting inference agent script...", flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use Gemma 3 4B (Ultra-lightweight, high-speed verified model)
API_BASE_URL = os.environ.get("API_BASE_URL", "google-gemini")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-3-4b-it")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("HF_TOKEN", "")

print(f"INFO: Configured with API_BASE_URL={API_BASE_URL}, MODEL_NAME={MODEL_NAME}", flush=True)

SYSTEM_PROMPT = """\
You are an expert customer support AI. You MUST follow this strict 3-step workflow.

OUTPUT FORMAT:
ONLY output a single JSON object. No extra text.

PHASE 1: CLASSIFICATION
Action: {"action_type": "classify", "payload": {"category": "...", "subcategory": "..."}}
Rule: NEVER ask questions or solve in Step 1.
Example: {"action_type": "classify", "payload": {"category": "billing", "subcategory": "unauthorized_charge"}}

PHASE 2: CLARIFICATION (If needed)
Action: {"action_type": "clarify", "payload": {"questions": ["..."]}}
Rule: Max 1 question. NEVER classify or resolve here.
Example: {"action_type": "clarify", "payload": {"questions": ["What is the transaction ID?"]}}

PHASE 3: RESOLUTION
Action: {"action_type": "resolve", "payload": {"steps": ["...", "..."], "message": "..."}}
Rule: Provide 4-6 real-world technical steps. 
Example: {"action_type": "resolve", "payload": {"steps": ["verify_id", "lookup_tx", "refund"], "message": "I have initiated the refund."}}

STRICT RULES:
1. Step 1 MUST be "classify".
2. After Step 1, NEVER use "classify".
3. If info is missing, use AT MOST ONE "clarify".
4. Otherwise, use "resolve".
5. NO extra conversation. ONLY JSON.
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
    # Improve JSON extraction for smaller models
    text = raw.strip()
    # Remove markdown code blocks
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    # Find the actual JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found: {raw[:100]}")
    
    json_str = text[start:end].strip()
    # Basic cleanup for small model quirks (like trailing commas before closing braces)
    json_str = json_str.replace(",\n}", "\n}").replace(",}", "}")
    
    data = json.loads(json_str)
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
        
        # V6 Hardened Query Logic with Retry & Rate-Limit Handling
        assistant_msg = ""
        for attempt in range(5):
            try:
                if "google" in API_BASE_URL or API_BASE_URL == "gemini" or API_BASE_URL == "google-gemini":
                    if not HAS_GENAI:
                        raise ImportError("google-genai not installed.")
                    client_gen = genai.Client(api_key=API_KEY)
                    try:
                        response_gen = client_gen.models.generate_content(
                            model=MODEL_NAME,
                            contents=user_prompt,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                temperature=0.0
                            )
                        )
                        assistant_msg = response_gen.text or ""
                    except Exception as sys_e:
                        if "instruction" in str(sys_e).lower():
                            # Fallback: Prepend system prompt to user prompt with clear markers
                            combined_prompt = f"### SYSTEM INSTRUCTIONS ###\n{SYSTEM_PROMPT}\n\n### CURRENT TICKET DATA ###\n{user_prompt}"
                            response_gen = client_gen.models.generate_content(
                                model=MODEL_NAME,
                                contents=combined_prompt,
                                config=genai.types.GenerateContentConfig(temperature=0.0)
                            )
                            assistant_msg = response_gen.text or ""
                        else:
                            raise sys_e
                else:
                    response_openai = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512,
                        timeout=45.0,
                    )
                    assistant_msg = response_openai.choices[0].message.content or ""
                
                if assistant_msg.strip():
                    break # Success
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "limit" in err_str:
                    wait_time = 5
                    print(f"  ⚠ Rate limited (429). Fast-retry in {wait_time}s (Attempt {attempt+1}/5)...")
                    time.sleep(wait_time)
                elif "404" in err_str or "not_found" in err_str:
                    # Specific fallbacks identified in diagnostic
                    fallbacks = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash-001", "gemini-1.5-flash"]
                    print(f"  ⚠ 404 error for {MODEL_NAME}. Trying diagnostic fallbacks...")
                    success = False
                    for alt_model in fallbacks:
                        if alt_model == MODEL_NAME: continue
                        print(f"  Attempting fallback to: {alt_model}...")
                        try:
                            if "google" in API_BASE_URL or API_BASE_URL == "gemini" or API_BASE_URL == "google-gemini":
                                try:
                                    response_gen = client_gen.models.generate_content(
                                        model=alt_model,
                                        contents=user_prompt,
                                        config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
                                    )
                                except:
                                    # Fallback for Gemma/etc
                                    response_gen = client_gen.models.generate_content(
                                        model=alt_model,
                                        contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                                        config=genai.types.GenerateContentConfig(temperature=0.0)
                                    )
                                assistant_msg = response_gen.text
                                success = True
                                break
                        except:
                            continue
                    if success: break
                    raise e
                else:
                    raise e
        
        # Minimal delay to respect shared infrastructure
        time.sleep(0.5)
            
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
    if not API_KEY:
        print("ERROR: Neither GEMINI_API_KEY nor HF_TOKEN is set.", file=sys.stderr)

    if not API_KEY:
        # Keep the process alive so the user can see the logs in HF Spaces
        import time
        while True:
            time.sleep(3600)

    client = None
    if not ("google" in API_BASE_URL or API_BASE_URL == "gemini" or API_BASE_URL == "google-gemini"):
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )

    results: list[dict[str, Any]] = []
    task_ids = list(SCENARIOS.keys())

    for i, task_id in enumerate(task_ids):
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

