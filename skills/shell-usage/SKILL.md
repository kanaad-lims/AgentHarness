---
name: shell-usage
version: 0.1.0
description: Run shell commands safely with user approval.
triggers: [bash, shell, terminal, command, run, execute, files, directory, list files]
tools: [bash]
category: software-development
---

# Shell Usage

Use `bash` to inspect the project: listing, reading, and searching files.

- Prefer small read-only commands; keep pipelines short.
- Never run delete, modify, network, or privileged commands unasked.
- Every command pauses for user approval — state what it does first.
- Never scrape the web with `bash`; use the search tools instead.
