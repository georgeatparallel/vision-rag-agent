"""
MCP server exposing web tools to the RAG agent.

Tools:
  - fetch_url     : fetch any URL as clean text/markdown
  - web_search    : DuckDuckGo free search (no API key)
  - wikipedia     : Wikipedia article search

Run standalone:  python rag_mcp_server.py
Used as subprocess by mcp_client.py via stdio transport.
"""
from __future__ import annotations

import json

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vision-rag-web-tools")


# ── fetch_url ──────────────────────────────────────────────────────────────

@mcp.tool()
def fetch_url(url: str, max_length: int = 8000) -> str:
    """Fetch the content of a URL and return it as plain text.

    Args:
        url: The URL to fetch.
        max_length: Maximum characters to return (default 8000).
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; vision-RAG-agent/1.0)"}
    try:
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")

        if "html" in content_type:
            try:
                from html.parser import HTMLParser

                class _TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self._parts: list[str] = []
                        self._skip = False

                    def handle_starttag(self, tag, attrs):
                        if tag in ("script", "style", "nav", "footer"):
                            self._skip = True

                    def handle_endtag(self, tag):
                        if tag in ("script", "style", "nav", "footer"):
                            self._skip = False

                    def handle_data(self, data):
                        if not self._skip:
                            stripped = data.strip()
                            if stripped:
                                self._parts.append(stripped)

                    def get_text(self) -> str:
                        return "\n".join(self._parts)

                extractor = _TextExtractor()
                extractor.feed(resp.text)
                text = extractor.get_text()
            except Exception:
                text = resp.text
        else:
            text = resp.text

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[truncated at {max_length} chars]"
        return text

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ── web_search ─────────────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str, max_results: int = 8) -> str:
    """Search the web using DuckDuckGo (free, no API key required).

    Returns a JSON list of results with title, snippet, and URL.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 8).
    """
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=10,
            follow_redirects=True,
        )
        data = resp.json()
        results: list[dict] = []

        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", "Abstract"),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                }
            )

        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": topic.get("Name", ""),
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", ""),
                    }
                )

        if not results:
            return json.dumps({"message": f"No results found for: {query}"})
        return json.dumps(results, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ── wikipedia ─────────────────────────────────────────────────────────────

@mcp.tool()
def wikipedia(topic: str, sentences: int = 5) -> str:
    """Fetch a Wikipedia summary for a topic.

    Args:
        topic: The topic to look up on Wikipedia.
        sentences: Number of summary sentences to return (default 5).
    """
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_"),
            timeout=10,
            follow_redirects=True,
        )
        if resp.status_code == 404:
            # Try search
            search = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "format": "json",
                    "srlimit": 1,
                },
                timeout=10,
            )
            hits = search.json().get("query", {}).get("search", [])
            if not hits:
                return f"No Wikipedia article found for: {topic}"
            title = hits[0]["title"]
            resp = httpx.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"),
                timeout=10,
                follow_redirects=True,
            )

        resp.raise_for_status()
        data = resp.json()
        summary = data.get("extract", "No summary available.")
        # Trim to requested sentences
        parts = summary.split(". ")
        trimmed = ". ".join(parts[:sentences])
        if not trimmed.endswith("."):
            trimmed += "."
        return f"**{data.get('title', topic)}**\n\n{trimmed}\n\nSource: {data.get('content_urls', {}).get('desktop', {}).get('page', '')}"

    except Exception as e:
        return f"Wikipedia error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
