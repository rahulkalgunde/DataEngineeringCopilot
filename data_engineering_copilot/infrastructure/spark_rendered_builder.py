"""Deterministic local rendering of pinned Spark documentation.

Runs the pinned Jekyll (``docs/``) and Sphinx (``python/docs/``) build
commands against the materialized Spark source tree, then enumerates the
rendered HTML output into a deterministic manifest. The builder never touches
Qdrant and never activates anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from data_engineering_copilot.config.settings import SparkRenderedBuildConfig, SparkRenderedSourceConfig

# Files produced by Sphinx/Jekyll that are never documentation content.
_NON_DOC_EXTENSIONS = frozenset({".js", ".css", ".json", ".png", ".jpg", ".gif", ".svg", ".ico", ".map"})


@dataclass(frozen=True)
class RenderedFileRecord:
    """A single rendered HTML output within a build output root."""

    build: str
    relative_path: str
    absolute_path: Path
    doc_type: str
    language: str
    canonical_url: str


@dataclass(frozen=True)
class RenderedManifest:
    """Deterministic enumeration of the rendered documentation outputs."""

    source_name: str
    ref: str
    commit: str
    root: Path
    files: tuple[RenderedFileRecord, ...]
    manifest_hash: str


@dataclass(frozen=True)
class RenderedBuildResult:
    """Outcome of a single rendered build execution."""

    build_name: str
    exit_code: int
    output_root: Path
    log_path: Path
    html_file_count: int


class SparkRenderedBuilder:
    """Build pinned Spark rendered documentation and produce a manifest.

    Parameters
    ----------
    config:
        Pinned Spark rendered documentation configuration.
    source_root:
        Materialized Spark source tree root (verified via ``.spark_commit``).
    artifact_root:
        Generation artifact directory that will hold the rendered outputs and
        build logs.
    python_executable:
        Python interpreter used for the Sphinx build. Defaults to the project
        ``dec_pydocs_venv`` interpreter when present, else ``sys.executable``.
    build_timeout:
        Per-command timeout in seconds.
    """

    def __init__(
        self,
        config: SparkRenderedSourceConfig,
        source_root: Path,
        artifact_root: Path,
        python_executable: Path | None = None,
        build_timeout: int = 1800,
    ) -> None:
        self._config = config
        self._source_root = source_root.resolve()
        self._artifact_root = artifact_root
        self._python_executable = python_executable
        self._build_timeout = build_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_source(self) -> Path:
        """Verify the materialized source marker and return the source root.

        Raises ``RuntimeError`` when the tree or its commit marker is missing
        or does not match the configured commit.
        """
        marker = self._source_root / ".spark_commit"
        if not self._source_root.exists():
            raise RuntimeError(f"Spark source root does not exist: {self._source_root}; run `dec spark-manifest` first")
        if not marker.is_file():
            raise RuntimeError(f"Spark source marker missing: {marker}; run `dec spark-manifest` first")
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded != self._config.commit:
            raise RuntimeError(
                f"Spark source marker {recorded!r} does not match configured commit "
                f"{self._config.commit!r}; re-run `dec spark-manifest`"
            )
        return self._source_root

    def render(self, log_name: str = "render_build.log") -> RenderedManifest:
        """Run every configured build and return the rendered manifest.

        Raises ``RuntimeError`` when any build command exits non-zero or an
        expected output root is missing after the build.
        """
        root = self.verify_source()
        log_path = self._artifact_root / log_name
        self._artifact_root.mkdir(parents=True, exist_ok=True)

        records: list[RenderedFileRecord] = []
        results: list[RenderedBuildResult] = []

        for build in self._config.builds:
            log_lines: list[str] = []
            output_root = self._output_root(build)
            self._wipe_output_root(output_root)
            command = self._build_command(build, output_root)
            env = self._build_env(build, root)

            log_lines.append(f"=== build {build.name} ===")
            log_lines.append(f"cwd: {self._source_root / build.working_dir}")
            log_lines.append("command: " + " ".join(command))
            log_lines.append("env: " + json.dumps(dict(env), sort_keys=True))
            log_lines.append("output_root: " + str(output_root))

            try:
                proc = subprocess.run(
                    command,
                    cwd=str(self._source_root / build.working_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self._build_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                log_lines.append(f"TIMEOUT after {self._build_timeout}s")
                self._append_log(log_path, log_lines)
                raise RuntimeError(f"Rendered build {build.name!r} timed out after {self._build_timeout}s") from exc
            except OSError as exc:
                log_lines.append(f"OSError: {exc}")
                self._append_log(log_path, log_lines)
                raise RuntimeError(f"Failed to launch rendered build {build.name!r}: {exc}") from exc

            log_lines.append(f"exit: {proc.returncode}")
            if proc.stdout:
                log_lines.append("--- stdout ---")
                log_lines.append(proc.stdout.rstrip())
            if proc.stderr:
                log_lines.append("--- stderr ---")
                log_lines.append(proc.stderr.rstrip())

            self._append_log(log_path, log_lines)

            if proc.returncode != 0:
                raise RuntimeError(f"Rendered build {build.name!r} exited {proc.returncode} (see {log_path})")

            if not output_root.exists() or not output_root.is_dir():
                raise RuntimeError(f"Rendered build {build.name!r} did not produce output root {output_root}")

            build_records = self._enumerate(build, output_root)
            records.extend(build_records)
            results.append(
                RenderedBuildResult(
                    build_name=build.name,
                    exit_code=proc.returncode,
                    output_root=output_root,
                    log_path=log_path,
                    html_file_count=len(build_records),
                )
            )

        canonical = {
            "source_name": self._config.name,
            "ref": self._config.ref,
            "commit": self._config.commit,
            "files": [
                {
                    "build": r.build,
                    "relative_path": r.relative_path,
                    "doc_type": r.doc_type,
                    "language": r.language,
                    "canonical_url": r.canonical_url,
                }
                for r in records
            ],
        }
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        manifest_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return RenderedManifest(
            source_name=self._config.name,
            ref=self._config.ref,
            commit=self._config.commit,
            root=root,
            files=tuple(records),
            manifest_hash=manifest_hash,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _output_root(self, build: SparkRenderedBuildConfig) -> Path:
        relative = build.output_root.replace("{output}", "output")
        base = self._artifact_root / build.name
        return (base / relative).resolve()

    def _build_command(self, build: SparkRenderedBuildConfig, output_root: Path) -> list[str]:
        if not build.command:
            raise RuntimeError(f"Rendered build {build.name!r} has an empty command")
        resolved: list[str] = []
        for part in build.command:
            part = part.replace("{output}", str(output_root))
            part = part.replace("{root}", str(self._source_root))
            part = part.replace("{python}", str(self._python_executable or _default_python()))
            resolved.append(part)
        return resolved

    def _build_env(self, build: SparkRenderedBuildConfig, root: Path) -> dict[str, str]:
        import os

        env = dict(os.environ)
        for key, value in build.env:
            env[key] = value.replace("{output}", str(self._output_root(build))).replace("{root}", str(root))
        return env

    def _wipe_output_root(self, output_root: Path) -> None:
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.parent.mkdir(parents=True, exist_ok=True)

    def _enumerate(self, build: SparkRenderedBuildConfig, output_root: Path) -> list[RenderedFileRecord]:
        records: list[RenderedFileRecord] = []
        for file_path in sorted(output_root.rglob("*")):
            if not file_path.is_file() or file_path.is_symlink():
                continue
            if file_path.suffix.lower() in _NON_DOC_EXTENSIONS:
                continue
            try:
                rel = file_path.relative_to(output_root).as_posix()
            except ValueError:
                continue
            if not self._matches(rel, build.include):
                continue
            if self._matches(rel, build.exclude):
                continue
            canonical_url = build.canonical_url.format(relpath=rel)
            records.append(
                RenderedFileRecord(
                    build=build.name,
                    relative_path=rel,
                    absolute_path=file_path,
                    doc_type=build.doc_type,
                    language=build.language,
                    canonical_url=canonical_url,
                )
            )
        return records

    @staticmethod
    def _matches(rel: str, patterns: tuple[str, ...]) -> bool:
        if not patterns:
            return False
        from data_engineering_copilot.infrastructure.spark_source_resolver import SparkSourceResolver

        return any(SparkSourceResolver._path_matches(rel, pattern) for pattern in patterns)

    @staticmethod
    def _write_log(log_path: Path, lines: list[str]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _append_log(log_path: Path, lines: list[str]) -> None:
        """Append a build block to the shared render log (append mode)."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def _default_python() -> Path:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    pydocs = project_root / "dec_pydocs_venv" / "bin" / "python"
    if pydocs.is_file():
        return pydocs
    return Path(sys.executable)


def load_rendered_manifest(path: Path, artifact_root: Path, config: SparkRenderedSourceConfig) -> RenderedManifest:
    """Load a persisted rendered manifest and reconstruct absolute file paths.

    The on-disk manifest stores each record relative to its build output root.
    ``artifact_root`` is the generation artifact directory (e.g.
    ``data/spark_corpus/<generation>``) and ``config`` maps build names to
    their output-root layout.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rendered manifest must be a JSON object")
    builds = {build.name: build for build in config.builds}
    files_raw = raw.get("files")
    if not isinstance(files_raw, list):
        raise ValueError("rendered manifest `files` must be a list")

    records: list[RenderedFileRecord] = []
    for entry in files_raw:
        build_name = entry.get("build")
        build = builds.get(build_name)
        if build is None:
            raise ValueError(f"rendered manifest references unknown build {build_name!r}")
        rel = entry["relative_path"]
        output_root_rel = build.output_root.replace("{output}", "output")
        absolute_path = (artifact_root / build.name / output_root_rel / rel).resolve()
        records.append(
            RenderedFileRecord(
                build=build.name,
                relative_path=rel,
                absolute_path=absolute_path,
                doc_type=entry.get("doc_type", build.doc_type),
                language=entry.get("language", build.language),
                canonical_url=entry["canonical_url"],
            )
        )

    return RenderedManifest(
        source_name=raw["source_name"],
        ref=raw["ref"],
        commit=raw["commit"],
        root=artifact_root,
        files=tuple(records),
        manifest_hash=raw["manifest_hash"],
    )
