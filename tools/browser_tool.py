"""Small, deterministic Playwright browser commands (no LLM involvement)."""

from urllib.parse import quote_plus, urlparse

from playwright.sync_api import sync_playwright


MAX_PAGE_TEXT = 5_000


def run_browser(command: str) -> str:
    """Run a supported browser command: ``open <url>`` or ``search <query>``."""
    command = command.strip()
    operation, _, value = command.partition(" ")
    operation = operation.lower()
    value = value.strip()

    if operation not in {"open", "search"} or not value:
        return (
            "Usage: /browser open <http(s) URL> or "
            "/browser search <search terms>"
        )

    if operation == "open":
        if urlparse(value).scheme not in {"http", "https"}:
            return "Error: URL must start with http:// or https://."
        url = value
    else:
        url = f"https://www.google.com/search?q={quote_plus(value)}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                page_text = page.locator("body").inner_text(timeout=10_000).strip()
                if len(page_text) > MAX_PAGE_TEXT:
                    page_text = page_text[:MAX_PAGE_TEXT] + "\n...[truncated]"

                return (
                    f"Title: {page.title()}\n"
                    f"URL: {page.url}\n\n"
                    f"{page_text or '(The page has no readable text.)'}"
                )
            finally:
                browser.close()
    except Exception as error:
        return f"Browser error: {error}"
