---
name: arxiv-research
version: 0.1.0
description: Find and cite arXiv papers for literature questions.
triggers: [arxiv, paper, papers, literature, publication, cite, citation]
tools: [arxiv_search]
category: research
---

# arXiv Research

Use `arxiv_search` for paper and literature questions only.

- Phrase queries with title or author wording when the user names them.
  Single-item requests go out with `max_results` 1 to get the full abstract.
- Broader surveys use a topic query and shortlist from the results.
- Cite the paper URL for every factual claim about a paper.
- If the results miss the target, say so and suggest a refined query.
  Never invent titles, authors, IDs, or URLs.
