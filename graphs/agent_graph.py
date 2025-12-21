# graphs/agent_graph.py
from __future__ import annotations

import os
import datetime as _dt
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

# ----------------------------
# Settings
# ----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
MAX_TOOL_LOOPS = int(os.getenv("MAX_TOOL_LOOPS", "3"))
TODAY_STR = _dt.date.today().isoformat()


def print_debug(title: str, content: str) -> None:
    if not DEBUG_MODE:
        return
    print(f"\n🐛 [DEBUG] {title}:\n{content}\n" + "-" * 40)


# ----------------------------
# Tools
# ----------------------------
def _format_tavily_results(tavily_response: object) -> str:
    if not isinstance(tavily_response, dict):
        return f"（検索結果の形式が想定外でした）\nraw: {tavily_response!r}"

    results = tavily_response.get("results", []) or []
    if not results:
        return "（検索結果なし）"

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip()
        content = (r.get("content") or r.get("raw_content") or "").strip()
        url = (r.get("url") or "").strip()

        # content が長すぎるとログ/プロンプトを圧迫するので軽く上限
        if len(content) > 900:
            content = content[:900] + "…"

        lines.append(f"[{i}] {title}\n{content}\nsource: {url}")

    return "\n\n".join(lines) if lines else "（検索結果なし）"


def _build_tools():
    # Tavily は任意: キー無しでもアプリ自体は動かす（検索はできない）
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_key:

        @tool
        def tavily_search_formatted(query: str) -> str:
            """Web検索（Tavily）。APIキー未設定の場合はその旨を返す。"""
            return (
                "（TAVILY_API_KEY が未設定のため Web検索を実行できません）\n"
                "`.env` に TAVILY_API_KEY を設定してください。"
            )

        return [tavily_search_formatted]

    from langchain_tavily import TavilySearch

    tavily_search = TavilySearch(
        max_results=3,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )

    @tool
    def tavily_search_formatted(query: str) -> str:
        """Web検索（Tavily）。上位結果を整形して返す。"""
        try:
            tavily_response = tavily_search.invoke({"query": query})
            return _format_tavily_results(tavily_response)
        except Exception as e:
            return f"（Tavily検索中にエラー）{type(e).__name__}: {e}"

    return [tavily_search_formatted]


tools = _build_tools()


# ----------------------------
# State
# ----------------------------
class State(TypedDict, total=False):
    # add_messages reducer: updateで渡したメッセージ配列が既存配列に追記される
    research_messages: Annotated[list[BaseMessage], add_messages]
    analysis_messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int

    # UI向け（開始時に立てる）
    current_step: str
    approval_decision: str

    # report_agent のみがセットする「最終レポート」
    final_report: str


# ----------------------------
# Model / Chains
# ----------------------------
model = ChatOpenAI(model=MODEL_NAME, temperature=0)
model_with_tools = model.bind_tools(tools)

research_prompt_text = f"""
あなたは事業リサーチ担当です。ユーザーのテーマについて、市場規模、主要プレイヤー、技術課題などをWeb検索で調査してください。
今日は {TODAY_STR} です。新しい情報が必要なら新しい情報を優先してください。
必要に応じて最適なツールを使用してください。

【出典ルール】
- ツール出力にある "source: URL" を根拠として使うときのみ、本文中に [n] を付けてください。
- 末尾に参照一覧（[n] URL）を付けてください。
- 存在しない出典は作らないこと。
""".strip()

research_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", research_prompt_text),
        MessagesPlaceholder(variable_name="research_messages"),
    ]
)
research_chain = research_prompt | model_with_tools

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは優秀な書記です。以下の「調査ログ」を要約し、市場分析チームが使える基礎レポートを作成してください。\n"
            "断定的な事実には可能な限り [n] を付け、末尾に参照一覧（[n] URL）を付けてください。存在しない出典は作らない。\n"
            "ツール出力に含まれる source: URL だけを参照として扱ってください。",
        ),
        ("human", "以下が調査ログです:"),
        MessagesPlaceholder(variable_name="research_messages"),
        ("human", "上記を元に、市場分析のための基礎レポートを作成してください。"),
    ]
)
summary_chain = summary_prompt | model

market_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは市場分析のプロです。レポートを元にSWOT分析を行ってください。"),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
market_chain = market_prompt | model

technical_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは技術のCTOです。市場分析を踏まえ、技術的課題と実現性を指摘してください。"),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
technical_chain = technical_prompt | model

report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "これまでの議論を統合し、投資家向けの具体的な事業プランを作成してください。\n"
            "文末での質問や提案は禁止です。「以上」で終わらせてください。",
        ),
        MessagesPlaceholder(variable_name="analysis_messages"),
    ]
)
report_chain = report_prompt | model


# ----------------------------
# Utility: start-marker nodes
# ----------------------------
def _mark_step(step: str, goto: str) -> Command:
    # 「開始時」に step を更新する軽量ノード
    return Command(update={"current_step": step}, goto=goto)


# ----------------------------
# Nodes
# ----------------------------
def research_start(state: State) -> Command[Literal["research_agent"]]:
    print_debug("Node", "research_start")
    return _mark_step("research_agent", "research_agent")


def research_agent(state: State) -> Command[Literal["tools_start", "summary_start"]]:
    print_debug("Node", "research_agent")

    response = research_chain.invoke({"research_messages": state.get("research_messages", [])})
    update = {"research_messages": [response]}
    current_count = state.get("loop_count", 0)

    # LLMがツール呼び出しを要求したら tools へ
    if getattr(response, "tool_calls", None):
        if current_count < MAX_TOOL_LOOPS:
            return Command(update=update, goto="tools_start")
        return Command(update=update, goto="summary_start")

    return Command(update=update, goto="summary_start")


def tools_start(state: State) -> Command[Literal["tools"]]:
    print_debug("Node", "tools_start")
    return _mark_step("tools", "tools")


tool_node = ToolNode(tools, messages_key="research_messages")


def research_tool_node(state: State) -> Command[Literal["research_start"]]:
    print_debug("Node", "tools")

    result = tool_node.invoke({"research_messages": state.get("research_messages", [])})

    last_message = result["research_messages"][-1]
    tool_text = last_message.content
    tool_text = tool_text if isinstance(tool_text, str) else str(tool_text)
    print_debug("Tool Output", tool_text[:300] + ("... (省略)" if len(tool_text) > 300 else ""))

    return Command(
        update={
            "research_messages": result["research_messages"],
            "loop_count": state.get("loop_count", 0) + 1,
        },
        goto="research_start",
    )


def summary_start(state: State) -> Command[Literal["summary_agent"]]:
    print_debug("Node", "summary_start")
    return _mark_step("summary_agent", "summary_agent")


def summary_agent(state: State) -> Command[Literal["market_start"]]:
    print_debug("Node", "summary_agent")
    response = summary_chain.invoke({"research_messages": state.get("research_messages", [])})
    return Command(
        update={"analysis_messages": [response], "loop_count": 0},
        goto="market_start",
    )


def market_start(state: State) -> Command[Literal["market_agent"]]:
    print_debug("Node", "market_start")
    return _mark_step("market_agent", "market_agent")


def market_agent(state: State) -> Command[Literal["technical_start"]]:
    print_debug("Node", "market_agent")
    response = market_chain.invoke({"analysis_messages": state.get("analysis_messages", [])})
    return Command(update={"analysis_messages": [response]}, goto="technical_start")


def technical_start(state: State) -> Command[Literal["technical_agent"]]:
    print_debug("Node", "technical_start")
    return _mark_step("technical_agent", "technical_agent")


def technical_agent(state: State) -> Command[Literal["human_approval_start"]]:
    print_debug("Node", "technical_agent")
    response = technical_chain.invoke({"analysis_messages": state.get("analysis_messages", [])})
    return Command(update={"analysis_messages": [response]}, goto="human_approval_start")


def human_approval_start(state: State) -> Command[Literal["human_approval"]]:
    print_debug("Node", "human_approval_start")
    return _mark_step("human_approval", "human_approval")


def _safe_preview_messages(messages: list[BaseMessage], limit: int = 3) -> list[dict]:
    out: list[dict] = []
    tail = messages[-limit:] if limit > 0 else messages
    for m in tail:
        content = m.content
        s = content if isinstance(content, str) else str(content)
        if len(s) > 1200:
            s = s[:1200] + "…"
        out.append({"type": type(m).__name__, "content": s})
    return out


def human_approval_node(
    state: State,
) -> Command[Literal["market_start", "report_start", "__end__"]]:
    """
    HITL（承認）ノード。
    - interrupt(payload) で確実に停止
    - resume された値は approval_decision に保存
    - decision が y 以外なら report へ行かない
    """
    print_debug("Node", "human_approval")

    payload = {
        "kind": "approval_request",
        "question": "ここまでの議論を承認してレポートを作成しますか？",
        "options": ["y", "n", "retry"],
        "analysis_preview": _safe_preview_messages(state.get("analysis_messages", []), limit=3),
    }

    user_decision = interrupt(payload)
    print_debug("Approval decision (raw)", repr(user_decision))

    if isinstance(user_decision, str):
        decision_str = user_decision.strip().lower()
    else:
        decision_str = str(user_decision).strip().lower()

    if decision_str not in ("y", "n", "retry"):
        decision_str = "n"

    update = {"approval_decision": decision_str}

    if decision_str == "y":
        return Command(update=update, goto="report_start")
    if decision_str == "retry":
        return Command(update=update, goto="market_start")
    return Command(update=update, goto=END)


def report_start(state: State) -> Command[Literal["report_agent"]]:
    print_debug("Node", "report_start")
    return _mark_step("report_agent", "report_agent")


def report_agent(state: State) -> Command[Literal["__end__"]]:
    print_debug("Node", "report_agent")

    if (state.get("approval_decision") or "").strip().lower() != "y":
        return Command(update={"final_report": ""}, goto=END)

    response = report_chain.invoke({"analysis_messages": state.get("analysis_messages", [])})
    text = response.content if isinstance(response.content, str) else str(response.content)

    return Command(update={"analysis_messages": [response], "final_report": text}, goto=END)


# ----------------------------
# Build Graph
# ----------------------------
builder = StateGraph(State)

builder.add_node("research_start", research_start)
builder.add_node("research_agent", research_agent)

builder.add_node("tools_start", tools_start)
builder.add_node("tools", research_tool_node)

builder.add_node("summary_start", summary_start)
builder.add_node("summary_agent", summary_agent)

builder.add_node("market_start", market_start)
builder.add_node("market_agent", market_agent)

builder.add_node("technical_start", technical_start)
builder.add_node("technical_agent", technical_agent)

builder.add_node("human_approval_start", human_approval_start)
builder.add_node("human_approval", human_approval_node)

builder.add_node("report_start", report_start)
builder.add_node("report_agent", report_agent)

builder.add_edge(START, "research_start")

# 以降は Command の goto で遷移（明示edgeは最小化）
builder.add_edge("report_agent", END)

graph = builder.compile()
