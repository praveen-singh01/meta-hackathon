import os
import sys
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import CustomerSupportEnv
from scenarios import SCENARIOS
from models import Action, ActionType, Observation, Reward
# Note: we need to import run_task, client, MODEL_NAME, etc. from inference
# or relocate them. For now, we'll import them.
import inference

app = FastAPI(title="OpenEnv Customer Support Agent")
global_env = CustomerSupportEnv()

class SolveRequest(BaseModel):
    task_id: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "online", "model": inference.MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset_env(request: Optional[SolveRequest] = None, task_id: Optional[str] = None):
    try:
        t_id = task_id or (request.task_id if request else None) or "easy_password_reset"
        obs = global_env.reset(t_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: Action):
    try:
        obs, reward, done, info = global_env.step(action)
        return {"observation": obs, "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solve")
def solve_task(request: Optional[SolveRequest] = None, task_id: Optional[str] = None, background_tasks: BackgroundTasks = None):
    t_id = task_id or (request.task_id if request else None) or "easy_password_reset"
    # Trigger the agent to solve the task and return logs
    client_local = inference.client
    background_tasks.add_task(inference.run_task, client_local, global_env, t_id)
    return {"status": "started", "task_id": t_id}

@app.get("/results")
def get_results():
    return {"results": "Detailed results available in Space logs."}

def main():
    print("INFO: Starting FastAPI server on port 7860...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
