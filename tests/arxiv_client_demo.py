import arxiv


QUERY = "Give papers on reinforcement learning in 2026"

search = arxiv.Search(
    query=QUERY,
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

client = arxiv.Client()
papers = list(client.results(search))

if not papers:
    print(f"No arXiv papers found for: {QUERY}")
else:
    for paper in papers:
        print(f"Title: {paper.title}")
        print(f"Published: {paper.published.date()}")
        print(f"URL: {paper.entry_id}")
        print(f"Abstract: {paper.summary.strip()}\n")
