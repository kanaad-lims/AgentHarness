<div align="center">
  <img src="images/harness_banner.png" alt="BOREAS-AGENT terminal demo" width="800" />
  <p><em>A modular, from-scratch coding-agent harness built with LangGraph and Python.</em></p>
  <p>Designed for tool-use loops, enforced policies, compact formatting, and a clean terminal experience.</p>
</div>

<div align="center">
  <a href="https://github.com/kanaad-lims/AgentHarness">
    <img src="https://img.shields.io/badge/GitHub-AgentHarness-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Repository" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Groq-FF6B35?style=flat-square&logo=groq&logoColor=white" alt="Groq SDK" />
  <img src="https://img.shields.io/badge/Rich-Textual-000000?style=flat-square&logo=textual&logoColor=white" alt="Rich Terminal UI" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/badge/Version-0.1.0-blue?style=flat-square" alt="Version 0.1.0" />
  <a href="https://deepwiki.com/kanaad-lims/AgentHarness">
    <img src="https://img.shields.io/badge/Docs-DeepWiki-4B32C3?style=flat-square&logo=readthedocs&logoColor=white" alt="DeepWiki Documentation" />
  </a>
</div>

<!-- <div align="center">
  <img src="images/harness_banner.png" alt="BOREAS-AGENT terminal demo" width="800" />
</div> -->

---

## What is this?

**BOREAS-AGENT** is an experimental coding agent harness — not a polished product, but a working foundation for exploring how LLMs can use tools reliably. It started as a curiosity project: *what does it take to make a model actually execute tasks, not just talk about them?*

The architecture is intentionally minimal and transparent:
- **LangGraph** drives the tool-calling loop — no hidden orchestration.
- **Groq** provides fast inference (current default: `qwen/qwen3.8-27b`), with the provider layer isolated so swapping models is straightforward.
- **Per-turn policies** enforce budgets, retries, and deduplication — so one runaway call can't eat your context or quota.
- **Compact presenters** trim tool output to signal, not noise — the model sees what matters.
- **Rich + prompt_toolkit** terminal UI — approvals, history, and tool traces stay readable.

> **Provider note:** Groq is the current default because of its speed and open-model access. Support for additional providers (OpenAI, Anthropic, local models via Ollama/vLLM, etc.) is planned — the `llm.py` abstraction makes this a configuration change, not a rewrite.

---

## How it works (the loop)

```text
┌─────────────────────────────────────────────────────────────────┐
│                        USER MESSAGE                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  agent.py — LangGraph StateGraph                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │   MODEL     │──▶   TOOLS      │──▶  POLICIES / CHECKS  │    │
│  │  (Groq)     │   │  (registry) │   │  (limits, retries)  │    │
│  └─────────────┘   └─────────────┘   └─────────────────────┘    │
│        ▲                                   │                    │
│        │           ┌───────────────────────┘──────────────────┐ │
│        └───────────│  PRESENTERS (compact formatting)         │ │
│                    └──────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │   FINAL ANSWER TO USER  │
              └─────────────────────────┘
```

**In words:**
1. Your message enters the graph with a fresh turn state.
2. The model (via `llm.py`) responds — either with an answer or `tool_calls`.
3. `run_tools` intercepts each call, runs policy checks (`policies/checks.py`), executes via `tools/registry.py`, formats results through `presenters/`, and returns tool messages.
4. The graph routes back to the model. This repeats until the model answers without tool calls or a limit is hit.
5. Observability (LangSmith + local JSONL) records every turn, model call, and tool execution — silently, without affecting the answer.

---

## Tools available today

| Tool | Purpose | Limits (per turn) | Notes |
|------|---------|-------------------|-------|
| `bash` | Run shell commands on the host (unsandboxed) | — | Requires `[y/N]` approval per call; 30s default timeout |
| `arxiv_search` | Search arXiv for papers | 3 calls, 10 results/call | Abstracts capped at 300 chars; formatted output ≤1,500 chars |
| `web_search` | DuckDuckGo general web search | 3 calls, 10 results/call | Snippets 300 chars; formatted output ≤1,500 chars |
| `write_todos` | Create a multi-step task plan | — | Exactly one `in_progress` item at a time |
| `read_skill` | Load a skill definition into context | 3 calls | Skill bodies truncated to 2,000 chars |
| `/browser` | Direct Playwright page open/search (bypasses model) | — | Manual command, not a model tool |

**Safeguards baked in:**
- **No global tool-call cap** — each tool has its own limit.
- **Max 8 model calls per turn** — hard stop with a graceful message.
- **Max 3 retries** for identical failed calls — then the model must change arguments.
- **Duplicate successes blocked** — the model sees the earlier result instead.
- **Result truncation** — every tool response clipped to 5,000 chars before returning.

---

## Skills (prompt-injection layer)

Skills are Markdown files with front-matter that load into the system prompt when triggered. Current skills:

| Skill | Triggers | Description |
|-------|----------|-------------|
| `arxiv-research` | paper, arxiv, literature, survey | Guidance for multi-paper arXiv workflows |
| `web-research` | search, web, browse, lookup | Heuristics for iterative web search |
| `task-planning` | plan, todo, breakdown, steps | Encourages explicit `write_todos` before acting |
| `system-design` | diagram, architecture, flowchart, design | ASCII system-design notation with one-line example trace |
| `shell-usage` | bash, command, shell, terminal | Safe shell patterns with approval reminders |

Skills live in `skills/<name>/SKILL.md`. No tools required — they're pure prompt guidance.

---

## Quick start

```bash
# 1. Clone & enter
git clone https://github.com/kanaad-lims/AgentHarness.git
cd AgentHarness

# 2. Create a virtual environment (conda recommended)
conda create -n boreas python=3.10 -y
conda activate boreas

# 3. Install deps
pip install -r requirements.txt
playwright install   # only if you want /browser

# 4. Configure
cp .env.example .env   # or create your own
# Edit .env and add:
# GROQ_API_KEY=your_key_here
# GROQ_MODEL=qwen/qwen3.8-27b   # optional

# 5. Run
python agent.py
```

**Inside the REPL:**
- Type naturally — the agent decides when to use tools.
- `/help` — show commands
- `/tools` — list available tools with schemas
- `/model` — show current model
- `/browser open <url>` — open a page directly (Playwright)
- `/bye` — exit

---

## Observability & evaluation

**LangSmith tracing** (optional): set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` (defaults to `boreas-agent`). Every turn becomes a trace with child runs for model calls and tool executions — failures never lose your answer.

**Local JSONL sink** (always on): every turn appends a full envelope to `runs/turns.jsonl` — turn ID, model calls, tool traces, verdicts, errors, metadata.

**In-house evals** (deterministic, free):
```bash
python -m evals.run runs/turns.jsonl
```
Metrics:
- `within_budgets` — token budgets respected
- `no_unknown_tools` — no phantom tool invocations
- `cites_sources_when_factual` — search-backed answers must include URLs
- `recovered_or_declared` — no silent failures or empty answers

Output includes pass rates + distributions (p50/p95 tokens, latencies, verdict counts).

---

## Project layout

```
.
├── agent.py              # LangGraph loop, dispatch, chat entry point
├── llm.py                # Groq client + single-request wrapper
├── config.py             # All settings & defaults (env-overridable)
├── requirements.txt
├── .env                  # Your keys (gitignored)
├── runs/                 # Local turn records (gitignored)
│   └── turns.jsonl
├── tools/
│   ├── registry.py       # Tool schemas + callables (model-ready)
│   ├── arxiv_search_tool.py
│   ├── web_search_tool.py
│   ├── bash_tool.py
│   ├── browser_tool.py
│   └── todo_tool.py
├── presenters/
│   ├── arxiv.py          # Compact arXiv formatting
│   ├── web.py            # Compact web formatting
│   └── bash.py           # Head/tail output formatting
├── policies/
│   ├── checks.py         # Pure policy predicates (unit-testable)
│   └── __init__.py       # Re-exports from config
├── prompts/
│   ├── messages.py       # Control strings, limit messages
│   └── __init__.py
├── cli/
│   ├── ui.py             # Rich console + approval cards
│   ├── prompt.py         # prompt_toolkit session
│   └── art.py            # ASCII banner
├── skills/
│   ├── loader.py         # Discovery, matching, injection
│   ├── skill_tool.py     # read_skill tool
│   └── <name>/SKILL.md   # Skill definitions
├── observability/
│   ├── recorder.py       # LangSmith + JSONL dual sink
│   └── turn.py           # Verdict mapping, metadata helpers
├── evals/
│   ├── metrics.py        # Deterministic code metrics
│   ├── run.py            # CLI runner
│   └── __init__.py
└── tests/
    ├── test_policies_checks.py
    ├── test_observability.py
    ├── test_evals.py
    └── ... (unit + golden-turn tests)
```

---

## Roadmap (what's next)

| Phase | Focus | Status |
|-------|-------|--------|
| **1. Tool-use tightening** | Remove global cap, per-tool limits, skill auto-loading, plan-first prompt |  Global cap removed; per-tool limits & retries active |
| **2. `/` command tools** | First-class slash commands for common workflows |  Planned |
| **3. Sandbox & permissions** | Filesystem/network/process isolation, explicit allow-lists |  Planned |
| **4. Context compaction** | Summarize, prune, re-rank history & tool output |  Planned |
| **5. Memory** | Short-term (session) + long-term (cross-session) persistence |  Planned |
| **6. Sub-agents** | Delegate parallel research/refactoring/verification |  Planned |
| **7. Evals & benchmarks** | BFCL → GAIA → WebArena/AssistantBench |  After harness stability |
| **8. Packaging** | `pipx install boreas-agent`, `boreas` CLI, API + UI |  Later |

---

## Contributing

Issues and PRs are welcome — especially around:
- Additional provider implementations in `llm.py`
- New skills in `skills/`
- Tool additions in `tools/`
- Policy refinements in `policies/checks.py`
- Eval metrics in `evals/metrics.py`

Run tests before submitting:
```bash
python -m pytest -q
```

---

## License

MIT — see [LICENSE](LICENSE).

---