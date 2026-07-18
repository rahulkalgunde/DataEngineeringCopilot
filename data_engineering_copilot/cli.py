from __future__ import annotations

import argparse
import json
import logging
import urllib.request

from data_engineering_copilot.config.logging import setup_logging
from data_engineering_copilot.config.settings import settings

logger = logging.getLogger(__name__)


def ingest(max_pages: int | None, source_names: tuple[str, ...] | None) -> None:
    from data_engineering_copilot.factory import build_ingestion_service

    logger.info("CLI ingest started max_pages=%s sources=%s", max_pages, source_names or "all")
    service = build_ingestion_service()
    total_chunks = service.ingest(max_pages_per_source=max_pages, source_names=source_names)
    logger.info("CLI ingest completed chunks=%s", total_chunks)
    print(f"Indexed {total_chunks} chunks.")


def ask(question: str) -> None:
    from data_engineering_copilot.factory import build_rag_service

    logger.info("CLI ask started question=%r", question[:200])
    service = build_rag_service()
    answer = service.answer(question)
    logger.info("CLI ask completed confidence=%.4f sources=%s", answer.confidence, len(answer.sources))
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"- {source.title}: {source.url}")
    print(f"\nConfidence: {answer.confidence:.2f}")


def reset_index() -> None:
    url = f"{settings.qdrant_url}/collections/{settings.collection_name}"
    logger.warning("Resetting Qdrant collection=%s url=%s", settings.collection_name, url)
    try:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            print(f"Deleted collection '{settings.collection_name}': {body}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"Collection '{settings.collection_name}' does not exist (nothing to reset).")
        else:
            raise
    logger.info("Qdrant collection reset completed collection=%s", settings.collection_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG assistant for data engineering documentation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Crawl documentation and build the QdrantDB index.")
    ingest_parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl per source.")
    ingest_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Documentation source name to ingest. Repeat to ingest multiple sources. Defaults to all sources.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the local repository.")
    ask_parser.add_argument("question", help="Question to answer.")

    subparsers.add_parser("reset-index", help="Delete the Qdrant collection so ingestion can rebuild it.")
    subparsers.add_parser("ui", help="Print the Streamlit command.")
    return parser


def main() -> None:
    if settings.logging_enabled:
        setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    logger.info("CLI command received command=%s", args.command)

    try:
        if args.command == "ingest":
            ingest(max_pages=args.max_pages, source_names=tuple(args.source) if args.source else None)
        elif args.command == "ask":
            ask(question=args.question)
        elif args.command == "reset-index":
            reset_index()
        elif args.command == "ui":
            logger.info("CLI ui command displayed Streamlit launch command")
            print("Run: python -m streamlit run data_engineering_copilot/ui/streamlit_app.py")
    except Exception:
        logger.exception("CLI command failed command=%s", args.command)
        raise


if __name__ == "__main__":
    main()
