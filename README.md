# BOREAS-AGENT

Experimental, from-scratch coding-agent harness in Python. A LangGraph
tool loop around a Groq-hosted model (configurable), with a small set of
tools, enforced per-turn policies, compact model-facing output formatting,
and a terminal UI. Version 0.1.0 — under active development.

## How it works

```
You → agent.py (LangGraph loop) → llm.py (one Groq request) → Groq
Groq → tool name + arguments → run_tools (policy checks) → registry
registry → raw tool + bounds + presenter → model-ready text → back to Groq
…repeat until the model answers with no further tool calls.
```

- `llm.py` owns provider communication only (native Groq SDK).
- `agent.py` owns orchestration: graph, state, dispatch, approvals, chat loop.
- `tools/` executes. `presenters/` formats results for the model.
- `policies/` enforces limits and failure/retry rules. `config.py` holds
  every setting. `prompts/` holds all text. `cli/` is the terminal UI.

## Tools

| Tool | What it does |
| --- | --- |
| `bash` | Runs shell commands (host, unsandboxed; every call needs `[y/N]` approval) |
| `arxiv_search` | Newest arXiv papers for a topic (up to 10/call, 3 calls/turn) |
| `web_search` | General web search via DuckDuckGo (up to 10/call, 3 calls/turn) |
| `write_todos` | Multi-deliverable task plans (exactly one `in_progress`) |
| `/browser` | Direct Playwright `open <url>` / `search <query>` (bypasses the model) |

Single-paper arXiv results return the full abstract; multi-paper lists
return shortened abstracts. Tool outputs are validated, budgeted, and
deduplicated; transient failures retry (max 3 identical attempts) while
repeated successes are blocked.

## Run it

```bash
pip install -r requirements.txt
playwright install            # only for /browser
```

Create a `.env` file:

```text
GROQ_API_KEY=<your key>
GROQ_MODEL=openai/gpt-oss-120b   # optional override, defaults in config.py
```

```bash
python agent.py
```

Commands inside the loop: `/help`, `/tools`, `/model`, `/browser …`, `/bye`.

## Test it

```bash
python -m pytest
```

Live-network tests are opt-in; the suite runs on fakes by default.

## Layout

```text
agent.py            LangGraph loop, dispatch, chat entry point
llm.py              Groq client + single-request call
config.py           All settings and defaults (env-overridable)
tools/              Tool implementations + registry.py (names, schemas, callables)
presenters/         Compact model-facing formatters (arxiv, web, bash)
policies/           Limits, retry/dedup rules (checks.py)
prompts/            System prompt + control messages
cli/                rich + prompt_toolkit terminal UI
tests/              Unit + golden-turn tests
```

## Roadmap

Skills + prompt-injection layer, sandboxing/permissions, context compaction,
short-term and long-term memory, subagents, layered verification, evals,
packaged install (`pipx install boreas-agent`, `boreas` command), API + UI.
