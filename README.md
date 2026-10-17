# Vision RAG Agent

**Agentic multimodal RAG** — ingests text, PDFs, and images into a vector database, then lets Claude autonomously decide which tools to call (local KB search, web fetch, Wikipedia, DuckDuckGo) before streaming a grounded answer.

---

## Architecture

```
Ingestion
  Text (.txt, .md, .rst, .pdf)
    ──▶ sentence-transformers (384-dim) ──▶ ChromaDB: text_docs

  Images (.png, .jpg, .webp, …)
    ──▶ Claude Vision  ──▶ caption ──▶ sentence-transformers ──▶ text_docs
    ──▶ CLIP (512-dim) ──────────────────────────────────────▶ image_docs

Agentic Query Loop
  user question
    ──▶ Claude (tool_use) ──▶ search_knowledge_base  ──▶ ChromaDB
                           ──▶ web_search             ──▶ DuckDuckGo (MCP)
                           ──▶ fetch_url              ──▶ live web page (MCP)
                           ──▶ wikipedia              ──▶ Wikipedia API (MCP)
                           ──▶ list_sources / ingest_file
    ──▶ loop until satisfied ──▶ stream final answer token by token
```

### Two embedding spaces

| Collection | Embedding model | Used for |
|---|---|---|
| `text_docs` | `all-MiniLM-L6-v2` (384-dim) | Text chunks + image captions |
| `image_docs` | CLIP `openai/clip-vit-base-patch32` (512-dim) | Visual cross-modal search |

Images are indexed **twice**: a CLIP embedding for visual similarity search, and a Claude-generated caption embedded for text-based retrieval.

---

## What makes it a real agent

| Pipeline RAG | This project |
|---|---|
| Fixed flow: retrieve → generate | Claude **decides** which tools to call and when |
| Single round-trip | Multi-step loop (up to 12 iterations) |
| KB only | KB + live web via **MCP** tools |
| No self-correction | Claude re-queries if results are weak |
| Text yielded at once | Final answer **streams token by token** |

---

## Tools available to Claude

### RAG tools (direct Python)
| Tool | What it does |
|---|---|
| `search_knowledge_base` | Vector search over text + image collections (sentence-transformers + CLIP) |
| `list_sources` | Show all ingested files and chunk counts |
| `ingest_file` | Add a local file to the KB mid-conversation |

### Web tools (via MCP — `rag_mcp_server.py`)
| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo search — free, no API key |
| `fetch_url` | Fetch any URL as clean text |
| `wikipedia` | Wikipedia article summary |

The MCP server runs as a **subprocess** via stdio transport using the `mcp` Python package. Tool schemas are loaded dynamically at agent startup.

### Optional Parallel web tools

Set `WEB_TOOLS_PROVIDER=parallel` in `.env` to use
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search and page fetching without a Parallel account or API key.
Search uses Fast mode. The client connects to `https://search.parallel.ai/mcp`
over Streamable HTTP; `wikipedia` continues to use the bundled stdio server.
Leave the setting unset, or set it to `local`, to use the original web tools.

The agent still sees `web_search` and `fetch_url`, with schemas discovered from
Parallel at startup. For direct Python calls, `web_search` takes `objective` and
`search_queries`; `fetch_url` maps to Parallel's `web_fetch` and takes `urls`.
Results are the server's text, including source URLs. Requests have a 60-second
overall timeout; call failures are returned as error text without switching providers.
Startup discovery failures stop tool loading. The anonymous service has usage
limits; see the linked setup guide. Claude queries still require your Anthropic key.

You can check the web tools independently of the embedding stack:

```bash
pip install -r requirements-web.txt
WEB_TOOLS_PROVIDER=parallel python - <<'PY'
from mcp_client import list_mcp_tools, call_mcp_tool

print([tool["name"] for tool in list_mcp_tools()])
print(call_mcp_tool("web_search", {
    "objective": "Find the Python documentation for virtual environments",
    "search_queries": ["Python venv documentation"],
}))
print(call_mcp_tool("fetch_url", {"urls": ["https://docs.python.org/3/library/venv.html"]}))
PY
```

Run the offline web-tool tests with `python -m unittest discover -s tests -v`
after installing `requirements-web.txt`.

---

## Quick start

### 1. Install dependencies

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# edit .env and add:  ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Ingest documents

```bash
# Ingest a whole folder (text, PDF, images)
python main.py ingest ./my_docs

# Or a single file
python main.py ingest path/to/report.pdf
python main.py ingest path/to/diagram.png
```

### 4. Query

```bash
# One-shot query
python main.py query "What brewing methods exist for coffee?"

# Verbose — see every tool call the agent makes
python main.py query "Who is the CEO of Anthropic?" --verbose

# Interactive chat
python main.py chat

# List all available tools
python main.py tools
```

---

## File structure

```
vision_RAG/
├── config.py            # Model names, paths, chunk settings
├── embeddings.py        # sentence-transformers + CLIP loaders
├── ingest.py            # Text/PDF/image ingestion pipeline
├── retriever.py         # Dual-collection ChromaDB retriever
├── tools.py             # RAG tool schemas + implementations
├── rag_mcp_server.py    # FastMCP server: fetch_url, web_search, wikipedia
├── mcp_client.py        # Stdio/optional Parallel HTTP MCP client wrapper
├── agent.py             # Agentic loop with true token streaming
├── main.py              # CLI: ingest / query / chat / tools
├── my_docs/             # Sample documents and images
└── requirements.txt
```

---

## Requirements

```
anthropic>=0.66.0
chromadb>=1.0.0
sentence-transformers>=3.0.0
transformers>=4.40.0
torch>=2.0.0
Pillow>=10.0.0
python-dotenv>=1.0.0
pypdf>=4.0.0
mcp>=1.26.0,<2
httpx>=0.27.1,<1
tqdm>=4.65.0
```

---

## Example verbose output

```
[agent] tools ready: [search_knowledge_base, list_sources, ingest_file,
                      fetch_url, web_search, wikipedia]

[agent] step 1 | stop=tool_use
  → search_knowledge_base({"query": "coffee brewing methods", "search_type": "both"})
     ↳ [{"source": "coffee_brewing_guide.txt", "distance": 0.42, ...}]
  → web_search({"query": "coffee brewing comparison"})
     ↳ {"message": "No results found"}

[agent] step 2 | stop=tool_use
  → search_knowledge_base({"query": "espresso AeroPress cold brew", "search_type": "text"})
     ↳ [{"source": "coffee_brewing_guide.txt", "distance": 0.27, ...}]

[agent] step 3 | stop=end_turn
# ☕ Coffee Brewing Methods ...   ← streamed token by token
```

---

## How image ingestion works

1. **CLIP embedding** — the raw image pixels are encoded by CLIP and stored in `image_docs` for visual cross-modal search
2. **Claude Vision caption** — Claude generates a rich text description of the image (subjects, colors, layout, visible text) stored in `text_docs` as `image_caption`
3. On query, **actual image bytes** are attached alongside search results so Claude can visually inspect retrieved images

---

## License

MIT
