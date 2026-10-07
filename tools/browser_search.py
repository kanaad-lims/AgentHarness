from ddgs import DDGS

def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Searches the internet using DuckDuckGo and returns a formatted text string
    of the top results for an LLM agent to read.
    """
    try:
        # Initialize the DuckDuckGo Search context manager
        with DDGS() as ddgs:
            # Fetch text results
            # region="wt-wt" means worldwide. You can change to "us-en", "in-en", etc.
            results = ddgs.text(query, region="wt-wt", max_results=max_results)
            
            if not results:
                return "No search results found."
            
            # Format the output into a clean string for the LLM
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No Title')
                href = result.get('href', 'No Link')
                body = result.get('body', 'No Description')
                
                formatted_results.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}\n")
                
            return "\n".join(formatted_results)
            
    except Exception as e:
        return f"An error occurred during the search: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    user_query = "Which indian freedom fighter quickly comes to mind when we say cellular jail? veer savarkar right?"
    print(f"Searching for: '{user_query}'...\n")
    
    search_output = web_search_tool(user_query, max_results=3)
    print(search_output)