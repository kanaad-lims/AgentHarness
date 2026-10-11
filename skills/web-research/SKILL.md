---
name: web-research
version: 0.1.0
description: Answer general web questions with cited search results.
triggers: [search, web, internet, latest, news, docs, documentation, who, when, current]
tools: [web_search]
category: research
---

# Web Research

Use `web_search` for general internet questions: current events, facts,
docs, and anything that is not a paper lookup.

- Answer ONLY from the returned results.
- Cite the source URL for every factual claim.
- If the results lack the answer, say so and suggest a refined query.
- Never invent titles, URLs, dates, or quotes.
