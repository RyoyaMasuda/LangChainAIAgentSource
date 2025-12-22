# backend/server.py
from __future__ import annotations

from typing import Optional, Literal, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from agent_graph import (
    new_thread_id,
    run_graph_start,
    run_graph_resume,
)

from contextlib import asynccontextmanager
from agent_graph import close_checkpointer

# ----------------------------
# FastAPI app
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_checkpointer()

app = FastAPI(title="LangServe + LangGraph HITL Agent", lifespan=lifespan)

# Live Server(例: http://127.0.0.1:5500) から叩くため、開発中はCORSを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Input/Output Schema
# ----------------------------
class AgentRequest(BaseModel):
    action: Literal["start", "resume"] = Field(..., description="start or resume")
    thread_id: Optional[str] = Field(None, description="同一スレッドでresumeするためのID")
    theme: Optional[str] = Field(None, description="start時のテーマ")
    decision: Optional[str] = Field(None, description="resume時の入力(y/n/retry)")


def agent_entry(req: Union[AgentRequest, dict]) -> dict:
    # LangServe の /invoke は { input: ... } 形式で受け取り、
    # runnable には input の中身（dict）が渡ってくる場合があるため、
    # dict のときは AgentRequest に詰め替えて属性アクセスできるようにする
    if isinstance(req, dict):
        req = AgentRequest(**req)

    # thread_id がなければ start 時に新規発行
    tid = req.thread_id or new_thread_id()

    print(f"[agent_entry] action={req.action} thread_id={tid}")

    if req.action == "start":
        theme = req.theme or "宇宙ゴミの回収事業"
        print(f"[agent_entry] start theme={theme}")  
        data = run_graph_start(theme=theme, thread_id=tid)
        print(f"[agent_entry] start done status={data.get('status')}")  
        return {"thread_id": tid, **data}

    # resume
    decision = (req.decision or "").strip().lower()
    print(f"[agent_entry] resume decision={decision}")  
    data = run_graph_resume(decision=decision, thread_id=tid)
    print(f"[agent_entry] resume done status={data.get('status')}")  
    return {"thread_id": tid, **data}


runnable = (
    RunnableLambda(agent_entry)
    # LangServe が OpenAPI 用のスキーマ生成をする際に入力型推論で落ちることがあるため明示
    .with_types(input_type=AgentRequest, output_type=dict)
)

# LangServe routes:
# POST /agent/invoke などが自動生成される
add_routes(app, runnable, path="/agent")

# run:
# uvicorn server:app --reload --port 8000