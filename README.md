<div align="center">

# ⛓️ LangChain Study

**Hand-written notebooks. No copy-paste solutions.**

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.x-1C3C3C?style=flat-square)](https://docs.langchain.com/oss/python/langchain/overview)
[![License](https://img.shields.io/badge/License-Personal-555?style=flat-square)]()

*Build muscle memory for LangChain v1 — from messages to agents, RAG, and LangGraph.*

[LangChain Docs](https://docs.langchain.com/oss/python/langchain/overview)

</div>

---

## What this is

A structured, notebook-by-notebook study path for **LangChain v1.0 – v1.3.15** (`>=1.0,<2`).

The goal is not to memorize APIs — it is to **write them from scratch**. Each lesson covers a mental model, core imports, hands-on tasks, and a memory checkpoint before moving on.

```text
messages → chat model → prompts → structured output
         → LCEL (|) → composition → tools → agents
         → RAG → memory → LangGraph
```

---

## Quick start

```bash
# clone & enter
cd langchain-study

# install (uv)
uv sync

# configure secrets — never commit .env
cp .env.example .env   # or create manually
```

```env
MODEL_PROVIDER=ollama          # or openai
MODEL=qwen3.5:9b               # or gpt-4o-mini
OPENAI_API_KEY=sk-...          # if using OpenAI
```

```bash
# launch notebooks
uv run jupyter lab
```

Open `00_setup.ipynb` and work through lessons in order. Pass each **checkpoint** before moving on.

---

## Curriculum

| # | Notebook | Topic | Status |
|:-:|----------|-------|:------:|
| 00 | [`00_setup.ipynb`](./00_setup.ipynb) | Setup & mental map | ✅ |
| 01 | [`01_messages.ipynb`](./01_messages.ipynb) | Messages | ✅ |
| 02 | [`02_chat_models.ipynb`](./02_chat_models.ipynb) | Chat models | ✅ |
| 03 | [`03_prompts.ipynb`](./03_prompts.ipynb) | Prompt templates | ✅ |
| 04 | [`04_structured_output.ipynb`](./04_structured_output.ipynb) | Structured output | ✅ |
| 05 | [`05_lcel.ipynb`](./05_lcel.ipynb) | Runnable & LCEL | ✅ |
| 06 | [`06_composition.ipynb`](./06_composition.ipynb) | Composition | 🚧 |
| 07 | `07_tools.ipynb` | Tools & manual loop | ⬜ |
| 08 | `08_agents.ipynb` | `create_agent` | ⬜ |
| 09 | `09_rag.ipynb` | RAG | ⬜ |
| 10 | `10_memory.ipynb` | Memory | ⬜ |
| 11 | `11_langgraph.ipynb` | LangGraph minimum | ⬜ |
| 99 | `99_from_memory.ipynb` | Final exam (from memory) | ⬜ |

---

## Core ideas

| Concept | One-liner |
|---------|-----------|
| **Messages** | Conversations are lists of typed message objects |
| **Runnable** | Every piece speaks the same protocol: `.invoke` / `.stream` / `.batch` |
| **LCEL** | `prompt \| model \| parser` — output flows left to right |
| **Structured output** | Pydantic schema in, validated object out |
| **Tools** | Model picks; **you** execute |
| **Agent** | Your tool loop, packaged as a graph |
| **RAG** | Retriever is just another Runnable |
| **Memory** | Message history or `thread_id` + checkpointer |

### Type flow (Lesson 05 checkpoint)

```text
dict  →  ChatPromptTemplate  →  messages
      →  ChatModel            →  AIMessage
      →  StrOutputParser      →  str
```

---

## Stack

- **LangChain** `>=1.0,<2` — core framework
- **langchain-ollama** — local models (e.g. Qwen via Ollama)
- **langchain-openai** — cloud models (optional)
- **python-dotenv** — `.env` config
- **uv** — dependency management

### Local model tips

- Set `reasoning=False` for thinking models (e.g. Qwen 3.5) to avoid long silent runs
- For Ollama structured output, prefer `method="function_calling"` over default `json_schema`

---

## What *not* to use

Pre-v1 patterns — do not copy these from old tutorials:

`LLMChain` · `AgentExecutor` · `from langchain.prompts import ...` · `ConversationBufferMemory`

Use **LCEL**, **`bind_tools`**, and **`create_agent`** instead.

---

## Project layout

```text
langchain-study/
├── 00_setup.ipynb …      # one notebook per lesson
├── README.md
├── pyproject.toml
└── .env                  # secrets (gitignored)
```

---

<div align="center">

**Read the lesson → write the notebook → pass the checkpoint → next.**

</div>
