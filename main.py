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


def export_index(output_path: str | None) -> None:
    import shutil
    from pathlib import Path

    out = Path(output_path) if output_path else Path("chroma_db_export.zip")
    logger.info("Exporting ChromaDB index from %s to %s", settings.chroma_dir, out)
    if not settings.chroma_dir.exists():
        print(f"ChromaDB directory does not exist: {settings.chroma_dir}")
        return
    # shutil.make_archive expects a base name without extension
    base = out.with_suffix("")
    archive = shutil.make_archive(str(base), 'zip', root_dir=str(settings.chroma_dir))
    print(f"Exported ChromaDB to: {archive}")


def import_index(archive_path: str) -> None:
    from pathlib import Path
    import zipfile

    archive = Path(archive_path)
    if not archive.exists():
        print(f"Archive not found: {archive}")
        return
    logger.info("Importing ChromaDB index from %s to %s", archive, settings.chroma_dir)
    # Remove existing dir first
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(archive), 'r') as zf:
        zf.extractall(path=str(settings.chroma_dir))
    print(f"Imported ChromaDB to: {settings.chroma_dir}")


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
    export_parser = subparsers.add_parser("export-index", help="Export the ChromaDB directory to a zip archive.")
    export_parser.add_argument("--output", help="Output zip path. Defaults to chroma_db_export.zip.")
    import_parser = subparsers.add_parser("import-index", help="Import a ChromaDB zip archive into local chroma_db.")
    import_parser.add_argument("archive", help="Path to the chroma_db zip archive to import.")
    subparsers.add_parser("ui", help="Print the Streamlit command.")
    return parser


def main() -> None:
    if settings.logging_enabled:
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
        elif args.command == "export-index":
            export_index(output_path=args.output)
        elif args.command == "import-index":
            import_index(args.archive)
        elif args.command == "ui":
            logger.info("CLI ui command displayed Streamlit launch command")
            print("Run: python -m streamlit run data_engineering_copilot/ui/streamlit_app.py")
    except Exception:
        logger.exception("CLI command failed command=%s", args.command)
        raise


if __name__ == "__main__":
    main()
