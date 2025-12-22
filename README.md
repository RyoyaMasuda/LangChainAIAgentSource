# PythonとLangChain/LangGraphで学ぶAIエージェント構築入門

書籍『PythonとLangChain/LangGraphで学ぶAIエージェント構築入門』の公式ソースコードリポジトリです。本書で解説している各ステップのソースコード（Jupyter NotebookおよびWebアプリ実装）を収録しています。

## 📘 本書の概要

本書は、Python、LangChain、およびLangGraphを用いて、自律的に思考し行動する「AIエージェント」を構築するための実践ガイドです。2025年リリースのLangChain 1.0 / LangGraph 1.0 世代のベストプラクティスに基づき、基礎からWebアプリへの展開までを網羅しています。

### 習得できる主なスキル

- OpenAI APIを使用したチャットボットの実装
- LangChainによるワークフロー（プロンプトテンプレート、ツール連携等）の構築
- LangGraphによる複雑なエージェントグラフと状態管理の実装
- 人間の承認（Human-in-the-Loop）を組み込んだエージェントの開発
- LangGraph Server（Agent Server）やLangServeを用いたWebアプリ化

## 📂 ディレクトリ構成

リポジトリ内の主要なファイルとフォルダの構成は以下の通りです。

### Jupyter Notebooks（基礎・学習用）

- `llm_basic.ipynb`: Chat Completions APIの基本とパラメータ
- `langchain_basic.ipynb`: LangChainの基本コンポーネント
- `langchain_chain.ipynb`: LCELを用いたチェーンの構築
- `langchain_webbot.ipynb`: Web検索連動型チャットボット
- `langchain_middle.ipynb`: ミドルウェアによる機能拡張（ガードレール等）
- `graph_basic.ipynb`: LangGraphの基本（ノード、エッジ、ステート）
- `graph_bot.ipynb`: LangGraphによるステートフルなチャットボット
- `graph_webbot.ipynb`: ツール呼び出しループを備えたエージェント
- `graph_ragbot.ipynb`: RAG（検索拡張生成）連動エージェント
- `aiagent.ipynb`: 本書メインの多機能AIエージェント実装

### Webアプリ実装（応用・運用用）

- `frontend/`: LangGraph ServerをバックエンドとしたWeb UI
- `frontend_ls/`: LangServeをバックエンドとしたWeb UI
- `graphs/`: LangGraph Serverにデプロイするグラフ定義
- `backend/`: LangServe用のサーバー実装
- `langgraph.json`: LangGraph Serverの設定ファイル


## 🚀 企業向け研修サービスのご案内

**Forest合同会社** では、生成AIを「知る」段階から「実業務で活用する」段階へつなげるための、企業向け実践研修を提供しています。

単なる知識の習得に留まらず、本リポジトリで扱うような **演習中心のカリキュラム** を通じて、自社で再現・応用できるスキルを最短距離で習得していただくことが特長です。

### 主な研修メニュー

- **生成AI入門ワークショップ**: プロンプトエンジニアリングの基礎から活用事例まで
- **AIエージェント構築ワークショップ**: 本書のトピックであるLangChain/LangGraphを用いた高度な開発手法
- **開発支援ツール導入研修**: GitHub CopilotやCursorを用いた次世代の開発スタイル
- **ノーコードAI開発**: Dify等を用いた、非エンジニアでも可能なAIアプリ開発

貴社の業務内容やエンジニアのスキルレベルに合わせたカスタマイズも可能です。無料でのオンライン相談も受け付けておりますので、ぜひお気軽にお問い合わせください。

🌐 **Forest合同会社 公式サイト**: [https://forest1.net/](https://forest1.net/)
