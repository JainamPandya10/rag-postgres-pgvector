"""
Usage:
    python -m src.cli ingest --path data/sample_docs
    python -m src.cli ask "What is pgvector used for?"
    python -m src.cli ask "What did Alice write about?" --author Alice --tags research
"""

import argparse
from tabulate import tabulate

from src.ingest import ingest_path
from src.retrieval import hybrid_search
from src.generate import generate_answer


def cmd_ingest(args):
    ingest_path(args.path, author=args.author, tags=args.tags)


def cmd_ask(args):
    chunks = hybrid_search(args.question, top_k=args.top_k, author=args.author, tags=args.tags)

    print("\nRetrieved chunks (ranked by hybrid RRF score):\n")
    table = [
        [i + 1, c.document_title, c.vector_rank, c.fulltext_rank, round(c.rrf_score, 4), c.content[:80] + "..."]
        for i, c in enumerate(chunks)
    ]
    print(tabulate(table, headers=["#", "Document", "VecRank", "FTRank", "RRF", "Preview"]))

    print("\n--- Answer ---\n")
    print(generate_answer(args.question, chunks))


def main():
    parser = argparse.ArgumentParser(description="RAG over Postgres + pgvector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Load documents into the database")
    p_ingest.add_argument("--path", required=True)
    p_ingest.add_argument("--author", default=None)
    p_ingest.add_argument("--tags", nargs="*", default=[])
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top_k", type=int, default=None)
    p_ask.add_argument("--author", default=None)
    p_ask.add_argument("--tags", nargs="*", default=[])
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
