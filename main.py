from __future__ import annotations

import argparse
import logging
import shutil

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.logging_config import configure_logging


logger = logging.getLogger("data_engineering_copilot.main")


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
    logger.warning("Resetting ChromaDB index path=%s", settings.chroma_dir)
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ChromaDB index reset path=%s", settings.chroma_dir)
    print(f"Reset ChromaDB index at: {settings.chroma_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RAG assistant for data engineering documentation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Crawl documentation and build the ChromaDB index.")
    ingest_parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to crawl per source.")
    ingest_parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Documentation source name to ingest. Repeat to ingest multiple sources. Defaults to all sources.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the local repository.")
    ask_parser.add_argument("question", help="Question to answer.")

    subparsers.add_parser("reset-index", help="Delete the local ChromaDB index so ingestion can rebuild it.")
    subparsers.add_parser("ui", help="Print the Streamlit command.")
    return parser


def main() -> None:
    configure_logging(settings.project_root)
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
            print(r"Run: .\.venv\Scripts\streamlit.exe run data_engineering_copilot\ui\streamlit_app.py")
    except Exception:
        logger.exception("CLI command failed command=%s", args.command)
        raise


if __name__ == "__main__":
    main()
