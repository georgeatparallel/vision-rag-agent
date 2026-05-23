"""
CLI entry point for the agentic multimodal RAG agent.

Usage:
  python main.py ingest <path>          # ingest a file or directory
  python main.py query  "<question>"    # one-shot agentic query
  python main.py chat                   # interactive REPL
  python main.py tools                  # list all available tools

Tools available to the agent:
  RAG (local):  search_knowledge_base, list_sources, ingest_file
  MCP (web):    fetch_url, web_search, wikipedia
"""
from __future__ import annotations

import sys


def cmd_ingest(path: str) -> None:
    from pathlib import Path
    from ingest import ingest_directory, ingest_single_file

    target = Path(path)
    if target.is_dir():
        ingest_directory(target)
    elif target.is_file():
        ingest_single_file(target)
    else:
        print(f"Error: '{path}' is not a valid file or directory.")
        sys.exit(1)


def cmd_tools() -> None:
    from agent import MultimodalRAGAgent
    agent = MultimodalRAGAgent()
    print("\nAvailable tools:")
    for t in agent._all_tools:
        print(f"  [{t['name']}]  {t['description'][:80]}")


def cmd_query(question: str, verbose: bool = False) -> None:
    from agent import MultimodalRAGAgent

    agent = MultimodalRAGAgent()
    print()
    for chunk in agent.query_stream(question, verbose=verbose):
        print(chunk, end="", flush=True)
    print("\n")


def cmd_chat(verbose: bool = False) -> None:
    from agent import MultimodalRAGAgent

    agent = MultimodalRAGAgent()
    print("Multimodal RAG — interactive mode  (type 'exit' or Ctrl-C to quit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        print("\nAssistant: ", end="", flush=True)
        for chunk in agent.query_stream(question, verbose=verbose):
            print(chunk, end="", flush=True)
        print("\n")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    command = args[0].lower()

    if command == "ingest":
        if len(args) < 2:
            print("Usage: python main.py ingest <file_or_directory>")
            sys.exit(1)
        cmd_ingest(args[1])

    elif command == "query":
        if len(args) < 2:
            print("Usage: python main.py query \"<your question>\"")
            sys.exit(1)
        cmd_query(" ".join(args[1:]), verbose=verbose)

    elif command == "chat":
        cmd_chat(verbose=verbose)

    elif command == "tools":
        cmd_tools()

    else:
        print(f"Unknown command: '{command}'\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
