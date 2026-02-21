"""
rag.py — Real-time web search RAG using duckduckgo-search DDGS directly.

Uses the `ddgs` package (formerly `duckduckgo-search`) DDGS class directly.
Result keys: 'title', 'body', 'href'

Public API:
  search(query) → (context_str, sources_list)
"""

from config import RAG_MAX_RESULTS


def search(query: str) -> tuple:
    """
    Perform a DuckDuckGo web search for `query`.

    Returns:
        (context_str, sources_list)
        - context_str : formatted text to inject into LLM system prompt,
                        or "" if search fails / returns nothing.
        - sources_list: list of {"title": ..., "link": ...} dicts for the UI,
                        or [] on failure.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=RAG_MAX_RESULTS))
    except Exception as e:
        print(f"[rag] Web search error: {e}")
        return "", []

    if not results:
        print(f"[rag] No results for: {query!r}")
        return "", []

    context_lines = [f"[Web search results for: '{query}']"]
    sources = []

    for i, r in enumerate(results, 1):
        title = (r.get("title", "") or "").strip()
        body  = (r.get("body",  "") or "").strip()   # v6 key (was 'snippet')
        href  = (r.get("href",  "") or "").strip()   # v6 key (was 'link')

        if body:
            context_lines.append(f"{i}. {title}: {body}")
        if href:
            sources.append({"title": title or href, "link": href})

    context = "\n".join(context_lines)
    print(f"[rag] Retrieved {len(results)} result(s) for: {query!r}")
    print(f"[rag] Context preview: {context[:200]}...")
    return context, sources
