# backend/agent_graph.py
from __future__ import annotations

import uuid
from typing import Annotated, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

load_dotenv("../.env")

MODEL_NAME = "gpt-5-mini"

DEBUG_MODE = True


def print_debug(title: str, content: str) -> None:
    if DEBUG_MODE:
        print(f"\n🐛 [DEBUG] {title}:\n{content}\n" + "-" * 40)


# ----------------------------
# Tools
# ----------------------------
tavily_search = TavilySearch(
    max_results=2,
    search_depth="basic",
    include_answer=False,
    include_raw_content=False,
    include_images=False,
)


def format_tavily_results(tavily_response: dict) -> str:
    results = tavily_response.get("results", [])
    if not results:
        return "（検索結果なし）"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        # contentが薄いケースに備えて raw_content も拾う（設定次第）
        content = r.get("content") or r.get("raw_content") or ""
        url = r.get("url", "")
        lines.append(f"[{i}] {title}\n{content}\nsource: {url}")
    return "\n\n".join(lines)


@tool
def tavily_search_formatted(query: str) -> str:
    """Web検索（Tavily）。上位結果を整形して返す。"""
    tavily_response = tavily_search.invoke({"query": query})
    return format_tavily_results(tavily_response)


tools = [tavily_search_formatted]


# ----------------------------
# State
# ----------------------------
class State(TypedDict):
    research_messages: Annotated[list[BaseMessage], add_messages]
    analysis_messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int


# ----------------------------
# Model
# ----------------------------
model = ChatOpenAI(model=MODEL_NAME)
model_with_tools = model.bind_tools(tools)


# ----------------------------
# Nodes
# ----------------------------
MAX_TOOL_LOOPS = 3

research_prompt_text = """
あなたは事業リサーチ担当です。ユーザーのテーマについて、市場規模、主要プレイヤー、技術課題などをWeb検索で調査してください。必要に応じて最適なツールを使用してください。
ツール呼び出し後に文章でまとめる場合は、ツール出力にある source: URL を根拠として使うときのみ [n] を付け、末尾に参照一覧を付けてください。存在しない出典は作らないこと。
"""

research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", research_prompt_text),
        MessagesPlaceholder(variable_name="research_messages"),
    ]
)
research_chain = research_prompt | model_with_tools


def research_agent(state: State) -> Command[Literal["tools", "summary_agent"]]:
    print_debug("Node", "research_agent")
    response = research_chain.invoke({"research_messages": state["research_messages"]})
    update = {"research_messages": [response]}
    current_count = state.get("loop_count", 0)

    if getattr(response, "tool_calls", None):
        if current_count < MAX_TOOL_LOOPS:
            return Command(update=update, goto="tools")
        return Command(update=update, goto="summary_agent")

    return Command(update=update, goto="summary_agent")


tool_node = ToolNode(tools, messages_key="research_messages")


def research_tool_node(state: State) -> Command[Literal["research_agent"]]:
    result = tool_node.invoke({"research_messages": state["research_messages"]})

    last_message = result["research_messages"][-1]
    tool_text = last_message.content
    tool_text = tool_text if isinstance(tool_text, str) else str(tool_text)
    print_debug("Tool Output", tool_text[:300] + "... (省略)")

    return Command(
        update={
            "research_messages": result["research_messages"],
            "loop_count": state.get("loop_count", 0) + 1,
        },
        goto="research_agent",
    )


summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは優秀な書記です。以下の「調査ログ」を要約し、市場分析チームが使える基礎レポートを作成してください。断定的な事実には可能な限り [n] を付け、末尾に参照一覧を付ける。存在しない出典は作らない。",
        ),
        ("human", "以下が調査ログです:"),
        MessagesPlaceholder(variable_name="research_messages"),
        ("human", "上記を元に、市場分析のための基礎レポートを作成してください。"),
    ]
)
summary_chain = summary_prompt | model


def summary_agent(state: State) -> dict:
    print_debug("Node", "summary_agent")
    response = summary_chain.invoke({"research_messages": state["research_messages"]})
    return {"analysis_messages": [response]}


market_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは市場分析のプロです。レポートを元にSWOT分析を行ってください。"),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
market_chain = market_prompt | model


def market_agent(state: State) -> dict:
    print_debug("Node", "market_agent")
    response = market_chain.invoke({"analysis_messages": state["analysis_messages"]})
    return {"analysis_messages": [response]}


technical_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは技術のCTOです。市場分析を踏まえ、技術的課題と実現性を指摘してください。",
        ),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
technical_chain = technical_prompt | model


def technical_agent(state: State) -> dict:
    print_debug("Node", "technical_agent")
    response = technical_chain.invoke({"analysis_messages": state["analysis_messages"]})
    return {"analysis_messages": [response]}


def human_approval_node(
    state: State,
) -> Command[Literal["market_agent", "report_agent", "__end__"]]:
    payload = {
        "kind": "approval_request",
        "question": "ここまでの議論を承認してレポートを作成しますか？",
        "options": ["y", "n", "retry"],
        "analysis_preview": [
            {
                "type": type(m).__name__,
                "content": (m.content[:500] + "…")
                if isinstance(m.content, str) and len(m.content) > 500
                else m.content,
            }
            for m in state.get("analysis_messages", [])
        ],
    }

    user_decision = interrupt(payload)

    if isinstance(user_decision, str):
        user_decision = user_decision.strip().lower()

    if user_decision == "y":
        return Command(goto="report_agent")
    if user_decision == "retry":
        return Command(goto="market_agent")
    return Command(goto=END)


report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
これまでの議論を統合し、投資家向けの具体的な事業プランを作成してください。
文末での質問や提案は禁止です。「以上」で終わらせてください。
""",
        ),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
report_chain = report_prompt | model


def report_agent(state: State) -> dict:
    print_debug("Node", "report_agent")
    response = report_chain.invoke({"analysis_messages": state["analysis_messages"]})
    return {"analysis_messages": [response]}


# ----------------------------
# Build Graph
# ----------------------------
builder = StateGraph(State)
builder.add_node("research_agent", research_agent)
builder.add_node("tools", research_tool_node)
builder.add_node("summary_agent", summary_agent)
builder.add_node("market_agent", market_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node("human_approval", human_approval_node)
builder.add_node("report_agent", report_agent)

builder.add_edge(START, "research_agent")
builder.add_edge("summary_agent", "market_agent")
builder.add_edge("market_agent", "technical_agent")
builder.add_edge("technical_agent", "human_approval")
builder.add_edge("report_agent", END)

# interrupt/resume を使うため checkpointer は必須（SQLite 永続化）
CHECKPOINT_DB_PATH = os.getenv("LANGGRAPH_CHECKPOINT_DB", "checkpoints.sqlite")

# NOTE: check_same_thread=False はOK（実装側でロックしてスレッド安全にする想定） :contentReference[oaicite:3]{index=3}
_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)

# 競合しやすい環境（同時アクセス）向けに推奨のPragma（任意だが実運用で効きやすい）
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute("PRAGMA synchronous=NORMAL;")
_conn.execute("PRAGMA busy_timeout=5000;")

checkpointer = SqliteSaver(_conn)
checkpointer.setup()  # テーブル作成などの初期化 :contentReference[oaicite:4]{index=4}

graph_app = builder.compile(checkpointer=checkpointer)


# ----------------------------
# API向けシリアライズ
# ----------------------------
def _as_text(m: object) -> str:
    if hasattr(m, "content"):
        c = getattr(m, "content")
        return c if isinstance(c, str) else str(c)
    return str(m)


def serialize_result(result: dict) -> dict:
    """
    LangGraphの戻り値にはメッセージオブジェクト等が含まれるため、
    APIレスポンスとしてJSON化できる形に変換する。
    """
    # interrupt の抽出（list/tuple/単体の差を吸収）
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
        payload = getattr(first, "value", first)
        return {
            "status": "interrupted",
            "interrupt": payload,
        }

    analysis = result.get("analysis_messages", [])
    analysis_serialized = [
        {"type": type(m).__name__, "content": _as_text(m)} for m in analysis
    ]

    report_text = analysis_serialized[-1]["content"] if analysis_serialized else ""

    return {
        "status": "completed",
        "report": report_text,
        "analysis_messages": analysis_serialized,
    }


def run_graph_start(theme: str, thread_id: str) -> dict:
    initial_state: dict = {
        "research_messages": [HumanMessage(content=f"テーマ: {theme}")],
        "loop_count": 0,
        "analysis_messages": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
    raw = graph_app.invoke(initial_state, config=config)
    return serialize_result(raw)


def run_graph_resume(decision: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    raw = graph_app.invoke(Command(resume=decision), config=config)
    return serialize_result(raw)


def new_thread_id() -> str:
    return str(uuid.uuid4())

def close_checkpointer() -> None:
    global _conn
    try:
        _conn.close()
    except Exception:
        pass