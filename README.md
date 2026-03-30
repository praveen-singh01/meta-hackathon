---
title: Customer Support OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support Simulation — OpenEnv Environment

> **An AI agent that doesn't just solve problems — it manages human emotions under pressure.**

This environment simulates **real-world high-stress customer interactions** where AI agents must not only resolve technical issues but also **de-escalate frustrated users**, **navigate ambiguous problems**, and **avoid taking shortcuts** — all while being scored on empathy, accuracy, and efficiency.

---

## Why This Matters

Every day, support teams handle thousands of tickets from customers who are confused, frustrated, or outright angry. The best agents don't just fix the problem — they **calm the customer down first**, then work through a structured diagnostic process. This environment captures that complexity:

- 📊 **Sentiment transitions**: Customers start frustrated or angry and react to the agent's tone
- 🎯 **Hidden ground truth**: The grader knows the real issue; the agent must discover it
- ⚖️ **Anti-gaming mechanics**: Diminishing returns, repeat penalties, and shortcut detection
- 🚫 **Bad path penalties**: Closing tickets prematurely, resolving wrong issues, or ignoring context

---

## Project Structure

```
hackathon/
├── models.py          # Pydantic models (Observation, Action, Reward, SentimentState)
├── scenarios.py       # Deterministic ticket scenarios + hidden ground truth
├── environment.py     # Core OpenEnv environment (step, reset, state)
├── graders.py         # 5 deterministic graders (classification, resolution, flow, sentiment, bad-path)
├── rewards.py         # Dense reward computation with anti-gaming mechanics
├── inference.py       # LLM agent loop (OpenAI-compatible)
├── openenv.yaml       # OpenEnv metadata & task definitions
├── Dockerfile         # Container packaging
├── requirements.txt   # Python dependencies
├── test_env.py        # 9 smoke tests
└── README.md
```

---

## Observation Space

| Field                  | Type             | Description                                         |
|------------------------|------------------|-----------------------------------------------------|
| `ticket`               | `TicketInfo`     | ID, customer details, subject, description           |
| `conversation_history` | `Message[]`      | Full chat log (customer / agent / system)            |
| `current_phase`        | `str`            | `classification`, `clarification`, or `resolution`   |
| `available_actions`    | `str[]`          | Valid action types for the current phase             |
| `sentiment.current`    | `str`            | `positive` / `neutral` / `frustrated` / `angry`      |
| `sentiment.frustration_score` | `float`   | 0.0 (calm) → 1.0 (maximum frustration)              |
| `sentiment.was_de_escalated` | `bool`     | Whether the agent successfully calmed the customer   |
| `urgency`              | `str`            | `low` / `medium` / `high`                            |
| `resolved`             | `bool`           | Whether the ticket has been resolved                 |
| `step_number`          | `int`            | Current step index                                   |
| `max_steps`            | `int`            | Maximum allowed steps                                |

---

## Action Space

| Action         | Payload                          | When to Use                          |
|----------------|----------------------------------|--------------------------------------|
| `classify`     | `{category, subcategory}`        | First step — identify the issue      |
| `clarify`      | `{question}`                     | Ambiguous issues — gather info       |
| `resolve`      | `{solution, steps: [...]}`       | Provide the fix with concrete steps  |
| `escalate`     | `{reason}`                       | Complex issues beyond agent scope    |
| `close_ticket` | `{}`                             | After confirmed resolution           |

---

## Reward Design

### Positive Rewards

| Component          | Weight | What It Measures                              |
|--------------------|--------|-----------------------------------------------|
| Classification     | +0.15  | Correct category & subcategory                |
| Clarification      | +0.15  | Relevant questions (with diminishing returns) |
| Resolution         | +0.30  | Coverage of gold-standard solution steps      |
| Flow / Closure     | +0.15  | Proper phase ordering & step efficiency       |
| Sentiment Mgmt     | +0.15  | Calming frustrated customers                  |
| Clean Behaviour    | +0.10  | No shortcuts or bad paths                     |

### Penalties

| Penalty                      | Value   | Trigger                                    |
|------------------------------|---------|--------------------------------------------|
| Repeat same action           | −0.1/ea | Consecutive identical action types         |
| Irrelevant clarification     | −0.05   | Question doesn't match required info       |
| Wrong classification         | −0.2    | Category mismatch                          |
| Premature closure            | −0.5    | Closing without resolving                  |
| Wrong-issue resolution       | −0.3    | Resolving with incorrect classification    |
| Skipped clarifications       | −0.2    | Resolving without asking required questions|

### Anti-Gaming: Diminishing Clarification Returns

| Clarification # | Reward Value |
|-----------------|-------------|
| 1st             | +0.20       |
| 2nd             | +0.10       |
| 3rd             | +0.05       |
| 4th+            | +0.02       |

---

## Tasks

| ID                       | Difficulty | Scenario                              | Initial Frustration |
|--------------------------|-----------|---------------------------------------|---------------------|
| `easy_password_reset`    | Easy      | Password reset — classify + resolve   | 30%                 |
| `medium_billing_dispute` | Medium    | Unclear charge — requires clarification| 40%                |
| `hard_data_migration`    | Hard      | Failed data import — multi-step reasoning| 70% (angry!)      |

---

## Hidden Ground Truth (Upgrade 3)

Each scenario has internal ground truth that the **agent never sees** but the **grader uses**:

```python
{
    "true_issue": "batch_import_timeout",
    "required_flow": ["classify", "clarify", "resolve"],
    "min_clarifications": 2,
    "max_clarifications": 4,
    "must_not_escalate": False,
    "must_de_escalate": True     # agent MUST calm the customer
}
```

This enables smarter evaluation without leaking answers.

---

## Setup

### Local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_env.py          # run 9 smoke tests (no API needed)
```

### Docker

```bash
docker build -t customer-support-env .
docker run \
  -e API_BASE_URL=<url> \
  -e MODEL_NAME=<model> \
  -e HF_TOKEN=<token> \
  customer-support-env
```

---

## Running

### Smoke Tests (no API required)

```bash
python test_env.py
```

### LLM Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

---

## Baseline Results

**Gold-standard (perfect) agent:**

| Task                     | Score | Notes                              |
|--------------------------|-------|------------------------------------|
| `easy_password_reset`    | ~0.95 | High score with empathy            |
| `medium_billing_dispute` | ~0.95 | Proper clarification + resolution  |
| `hard_data_migration`    | ~0.95 | De-escalation + multi-step solve   |

**Bad agent (wrong classification, no empathy):**

| Task                     | Score | Notes                              |
|--------------------------|-------|------------------------------------|
| `easy_password_reset`    | ~0.30 | Penalised for bad path + sentiment |

---

## OpenEnv Compliance

```python
env = CustomerSupportEnv()
obs = env.reset("easy_password_reset")       # → Observation
obs, reward, done, info = env.step(action)   # → (Observation, Reward, bool, dict)
obs = env.state()                            # → Observation (read-only)
```

All grading is **fully deterministic** — zero randomness.
