"""Pinned GitHub source resolver.

Acquires a pinned release tarball (by commit SHA) and builds a deterministic
manifest of files selected by the configured streams. Uses only the Python
standard library — no Git, no GitHub API, no third-party client.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from data_engineering_copilot.domain.protocols import GitRepoSource

# Safety cap for individual source files; larger files are skipped (in bytes).
_DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024

# Known binary/asset extensions excluded from all streams.
_BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".jar",
        ".class",
        ".parquet",
        ".avro",
        ".bin",
        ".so",
        ".dll",
        ".whl",
    }
)

_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".scala": "scala",
    ".java": "java",
    ".sql": "sql",
    ".r": "r",
    ".md": "conceptual",
    ".markdown": "conceptual",
    ".rst": "conceptual",
    ".txt": "conceptual",
}


@dataclass(frozen=True)
class SparkFileRecord:
    """A single selected file within the materialized Spark source tree."""

    stream: str
    relative_path: str
    absolute_path: Path
    doc_type: str
    language: str
    source_url: str


@dataclass(frozen=True)
class SparkManifest:
    """Deterministic enumeration of the files selected from a Spark release."""

    source_name: str
    ref: str
    commit: str
    root: Path
    files: tuple[SparkFileRecord, ...]
    manifest_hash: str


class SparkSourceResolver:
    """Materialize a pinned Spark release and build a file manifest.

    Parameters
    ----------
    config:
        Pinned Spark release configuration.
    cache_dir:
        Directory in which materialized source trees are cached by commit.
    max_file_bytes:
        Maximum file size in bytes accepted into the manifest.
    """

    def __init__(
        self,
        config: GitRepoSource,
        cache_dir: Path,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir
        self._max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self) -> SparkManifest:
        """Materialize (or reuse) the pinned source and return its manifest."""
        root = self.materialize()
        return self.build_manifest(root)

    def materialize(self) -> Path:
        """Ensure the pinned release is present on disk and return its root.

        Returns the cached root path. Raises ``RuntimeError`` for acquisition
        or integrity failures.
        """
        root = self._cache_root()
        marker = root / ".spark_commit"
        if root.exists() and marker.exists() and marker.read_text(encoding="utf-8").strip() == self._config.commit:
            return root

        if root.exists():
            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)

        archive_url = self._archive_url()
        temp_dir = Path(tempfile.mkdtemp(prefix="spark-src-", dir=str(self._cache_dir)))
        try:
            archive_path = temp_dir / "spark.tar.gz"
            self._download(archive_url, archive_path)
            extract_root = temp_dir / "extract"
            extract_root.mkdir()
            self._safe_extract(archive_path, extract_root)
            extracted = self._single_top_level_dir(extract_root)
            if extracted is None:
                raise RuntimeError(f"Archive did not contain a single top-level directory: {archive_url}")
            shutil.move(str(extracted), str(root))
            marker.write_text(self._config.commit, encoding="utf-8")
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Failed to materialize Spark source {self._config.ref}: {exc}") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return root

    def build_manifest(self, root: Path) -> SparkManifest:
        """Enumerate matching files and build a deterministic manifest."""
        resolved_root = root.resolve()
        if not resolved_root.exists():
            raise ValueError(f"Spark source root does not exist: {resolved_root}")

        records: list[SparkFileRecord] = []
        for stream in self._config.streams:
            include_patterns = list(stream.include)
            exclude_patterns = list(stream.exclude)
            for file_path in sorted(resolved_root.rglob("*")):
                if not file_path.is_file() or file_path.is_symlink():
                    continue
                rel = self._safe_relative(resolved_root, file_path)
                if rel is None:
                    continue
                if self._is_excluded(rel, exclude_patterns):
                    continue
                if not self._matches_include(rel, include_patterns):
                    continue
                if not self._matches_content(file_path, stream.content_requires):
                    continue
                if file_path.stat().st_size > self._max_file_bytes:
                    continue
                if file_path.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                language = self._language_for(file_path, stream)
                records.append(
                    SparkFileRecord(
                        stream=stream.name,
                        relative_path=rel,
                        absolute_path=file_path,
                        doc_type=stream.doc_type,
                        language=language,
                        source_url=self._source_url_for(rel),
                    )
                )

        canonical = {
            "source_name": self._config.name,
            "ref": self._config.ref,
            "commit": self._config.commit,
            "files": [
                {
                    "stream": r.stream,
                    "relative_path": r.relative_path,
                    "doc_type": r.doc_type,
                    "language": r.language,
                }
                for r in records
            ],
        }
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        manifest_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return SparkManifest(
            source_name=self._config.name,
            ref=self._config.ref,
            commit=self._config.commit,
            root=resolved_root,
            files=tuple(records),
            manifest_hash=manifest_hash,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_root(self) -> Path:
        digest = hashlib.sha256(self._config.commit.encode("ascii")).hexdigest()[:16]
        return self._cache_dir / f"{self._config.ref.replace('/', '_')}-{digest}"

    def _archive_url(self) -> str:
        owner, name = self._repo_parts()
        return f"https://github.com/{owner}/{name}/archive/{self._config.commit}.tar.gz"

    def _repo_parts(self) -> tuple[str, str]:
        """Return ``(owner, name)`` from the configured GitHub repository URL."""
        repo = self._config.repository.rstrip("/")
        if repo.endswith(".git"):
            repo = repo[:-4]
        if "github.com" not in repo:
            raise RuntimeError(f"Unsupported repository: {self._config.repository!r}")
        parts = repo.split("/")
        if len(parts) < 2:
            raise RuntimeError(f"Unsupported repository: {self._config.repository!r}")
        return parts[-2], parts[-1]

    def _download(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "data-engineering-copilot/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)

    def _safe_extract(self, archive_path: Path, target: Path) -> None:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Unsafe path in archive: {member.name!r}")
            tar.extractall(target)

    @staticmethod
    def _single_top_level_dir(root: Path) -> Path | None:
        entries = [p for p in root.iterdir() if p.is_dir()]
        if len(entries) == 1:
            return entries[0]
        return None

    @staticmethod
    def _safe_relative(root: Path, path: Path) -> str | None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            return None
        parts = rel.parts
        if any(part == ".." for part in parts):
            return None
        return rel.as_posix()

    @staticmethod
    def _is_excluded(rel: str, exclude_patterns: list[str]) -> bool:
        return any(SparkSourceResolver._path_matches(rel, pattern) for pattern in exclude_patterns)

    @staticmethod
    def _matches_content(file_path: Path, content_requires: tuple[str, ...]) -> bool:
        """Return True when the file contains every required substring.

        An empty ``content_requires`` passes every file. Content reads are
        bounded by ``_DEFAULT_MAX_FILE_BYTES`` so an oversize file is rejected
        before any read happens (callers check size first).
        """
        if not content_requires:
            return True
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return all(needle in text for needle in content_requires)

    @staticmethod
    def _matches_include(rel: str, include_patterns: list[str]) -> bool:
        return any(SparkSourceResolver._path_matches(rel, pattern) for pattern in include_patterns)

    @staticmethod
    def _path_matches(rel: str, pattern: str) -> bool:
        """Match a POSIX relative path against a glob with ``**`` recursion.

        Converts ``**`` segments into the equivalent regex so that leading,
        trailing, and mid-path ``**`` all work (e.g. ``**/tests/**``).
        """
        import re as _re

        posix = Path(rel).as_posix()
        if pattern == "**":
            return True
        if "**" in pattern:
            # Translate the glob to a regex. ``**/`` means "any number of
            # leading directories"; a bare ``**`` means "anything".
            regex = pattern
            regex = _re.escape(regex)
            regex = regex.replace(r"\*\*/", "(?:[^/]+/)*")
            regex = regex.replace(r"\*\*", ".*")
            regex = regex.replace(r"\*", "[^/]*")
            regex = regex.replace(r"\?", "[^/]")
            return _re.fullmatch(regex, posix) is not None
        return fnmatch.fnmatch(posix, pattern)

    def _language_for(self, file_path: Path, stream) -> str:
        if stream.language != "mixed" and stream.language != "conceptual":
            return stream.language
        return _LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), "conceptual")

    def _source_url_for(self, rel: str) -> str:
        owner, name = self._repo_parts()
        return f"https://raw.githubusercontent.com/{owner}/{name}/{self._config.commit}/{rel}"
