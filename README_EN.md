# Introduction to AI Agent Construction with Python, LangChain, and LangGraph

This repository contains the official source code for the book *"Introduction to AI Agent Construction with Python, LangChain, and LangGraph"* (published in 2025). It provides step-by-step implementations ranging from basic LLM calls to advanced autonomous agents with Web UIs.

## 📘 About the Book

This book is a practical guide to building "AI Agents"—systems that can perceive their environment and act autonomously to achieve goals. It focuses on the latest best practices using **LangChain 1.0** and **LangGraph 1.0**.

### Key Learning Outcomes

- Implement AI chatbots using OpenAI's Chat Completions API.
- Build structured workflows using LangChain (LCEL, Prompt Templates, Tool Integration).
- Develop complex agentic loops and state management with LangGraph.
- Integrate **Human-in-the-Loop (HITL)** for manual approval and intervention.
- Deploy agents as Web applications using **LangGraph Server** and **LangServe**.

## 📂 Repository Structure

### Jupyter Notebooks (Core Concepts)

- `llm_basic.ipynb`: Basics of Chat Completions API and parameters.
- `langchain_basic.ipynb`: Introduction to LangChain components.
- `langchain_chain.ipynb`: Building workflows with LCEL.
- `langchain_webbot.ipynb`: Chatbot integrated with Web Search.
- `langchain_middle.ipynb`: Extending functionality with Middleware (e.g., Guardrails).
- `graph_basic.ipynb`: LangGraph fundamentals (Nodes, Edges, State).
- `graph_bot.ipynb`: Stateful chatbots using LangGraph.
- `graph_webbot.ipynb`: Agents with tool-calling loops.
- `graph_ragbot.ipynb`: RAG (Retrieval-Augmented Generation) integrated agents.
- `aiagent.ipynb`: The main "Business Plan Generation Agent" featuring multi-agent collaboration.

### Web Application Implementations

- `frontend/`: Modern Web UI using **LangGraph Server** as the backend.
- `frontend_ls/`: Web UI using **LangServe** as the backend.
- `graphs/`: Graph definitions for LangGraph Server.
- `backend/`: Server implementation for LangServe.
- `langgraph.json`: Configuration for LangGraph Server.

## 🚀 Corporate Training Services by Forest, LLC

**Forest, LLC** provides professional hands-on training for companies looking to transition from "knowing AI" to "implementing AI in business."

We offer customized workshops based on the latest AI agent technologies featured in this repository. Our curriculum is designed to help your team build production-ready AI systems tailored to your specific business needs.

### Training Menus

* **Generative AI Fundamentals**: From prompt engineering to business use cases.
* **AI Agent Development**: Advanced implementation using LangChain and LangGraph.
* **AI-Driven Development**: Modernizing workflows with GitHub Copilot and Cursor.
* **No-code AI Solutions**: Building AI apps with platforms like Dify.

We offer online consultations to discuss your team's specific requirements.

🌐 **Official Website**: [https://forest1.net/](https://forest1.net/)