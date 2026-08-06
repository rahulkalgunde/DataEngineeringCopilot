"""Task 4 tests: SparkRenderedBuilder — offline builds, manifests, failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.config.settings import (
    SparkRenderedBuildConfig,
    SparkRenderedSourceConfig,
    load_spark_rendered_source_config,
)
from data_engineering_copilot.infrastructure.spark_rendered_builder import SparkRenderedBuilder

_COMMIT = "fa33ea000a0bda9e5a3fa1af98e8e85b8cc5e4d4"


def _build_config(name: str = "fake_build", **overrides) -> SparkRenderedBuildConfig:
    values = {
        "name": name,
        "doc_type": "guide",
        "language": "conceptual",
        "working_dir": ".",
        "command": (
            "{python}",
            "-c",
            "import pathlib; pathlib.Path('{output}').mkdir(parents=True, exist_ok=True); pathlib.Path('{output}/out.html').write_text('<html>')",
        ),
        "env": (("SKIP_API", "1"),),
        "output_root": "{output}",
        "include": ("**/*.html",),
        "exclude": (),
        "content_root_selector": "div#content",
        "excluded_selectors": (),
        "canonical_url": "https://spark.apache.org/docs/4.0.0/{relpath}",
        "renderer": "fake",
    }
    values.update(overrides)
    return SparkRenderedBuildConfig(**values)


def _rendered_config(*builds: SparkRenderedBuildConfig) -> SparkRenderedSourceConfig:
    return SparkRenderedSourceConfig(
        name="Apache Spark 4.0.0",
        repository="https://github.com/apache/spark.git",
        ref="v4.0.0",
        commit=_COMMIT,
        license="Apache-2.0",
        builds=builds,
    )


def _make_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "spark-src"
    root.mkdir(parents=True)
    (root / ".spark_commit").write_text(_COMMIT, encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
    return root


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def test_load_rendered_config_valid(tmp_path) -> None:
    config = {
        "name": "Apache Spark 4.0.0",
        "repository": "https://github.com/apache/spark.git",
        "ref": "v4.0.0",
        "commit": _COMMIT,
        "license": "Apache-2.0",
        "builds": [
            {
                "name": "jekyll_docs",
                "doc_type": "guide",
                "language": "conceptual",
                "working_dir": "docs",
                "command": ["bundle", "exec", "jekyll", "build"],
                "env": {"SKIP_API": "1"},
                "output_root": "{output}",
                "include": ["**/*.html"],
                "exclude": [],
                "content_root_selector": "div#content",
                "excluded_selectors": [],
                "canonical_url": "https://spark.apache.org/docs/4.0.0/{relpath}",
                "renderer": "jekyll",
            }
        ],
    }
    import json

    path = tmp_path / "spark_rendered_sources.json"
    path.write_text(json.dumps(config))
    loaded = load_spark_rendered_source_config(path)
    assert loaded.commit == _COMMIT
    assert loaded.builds[0].name == "jekyll_docs"
    assert loaded.builds[0].env == (("SKIP_API", "1"),)


def test_load_rendered_config_bad_commit_raises(tmp_path) -> None:
    config = {
        "name": "Apache Spark 4.0.0",
        "repository": "https://github.com/apache/spark.git",
        "ref": "v4.0.0",
        "commit": "nope",
        "license": "Apache-2.0",
        "builds": [
            {
                "name": "b",
                "doc_type": "guide",
                "language": "conceptual",
                "working_dir": "docs",
                "command": ["x"],
                "env": {},
                "output_root": "{output}",
                "include": ["**/*.html"],
                "exclude": [],
                "content_root_selector": "div#content",
                "excluded_selectors": [],
                "canonical_url": "u",
                "renderer": "jekyll",
            }
        ],
    }
    import json

    path = tmp_path / "spark_rendered_sources.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="commit"):
        load_spark_rendered_source_config(path)


def test_load_rendered_config_empty_builds_raises(tmp_path) -> None:
    config = {
        "name": "Apache Spark 4.0.0",
        "repository": "https://github.com/apache/spark.git",
        "ref": "v4.0.0",
        "commit": _COMMIT,
        "license": "Apache-2.0",
        "builds": [],
    }
    import json

    path = tmp_path / "spark_rendered_sources.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="builds"):
        load_spark_rendered_source_config(path)


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------


def test_verify_source_requires_marker(tmp_path) -> None:
    source_root = tmp_path / "missing"
    source_root.mkdir()
    builder = SparkRenderedBuilder(
        _rendered_config(_build_config()),
        source_root,
        tmp_path / "artifact",
    )
    with pytest.raises(RuntimeError, match="marker"):
        builder.verify_source()


def test_verify_source_requires_commit_match(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    (source_root / ".spark_commit").write_text("deadbeef" * 5, encoding="utf-8")
    builder = SparkRenderedBuilder(
        _rendered_config(_build_config()),
        source_root,
        tmp_path / "artifact",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        builder.verify_source()


def test_render_success_writes_manifest_and_log(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    artifact_root = tmp_path / "artifact"
    builder = SparkRenderedBuilder(
        _rendered_config(_build_config()),
        source_root,
        artifact_root,
    )
    manifest = builder.render()
    assert len(manifest.files) == 1
    record = manifest.files[0]
    assert record.relative_path == "out.html"
    assert record.canonical_url == "https://spark.apache.org/docs/4.0.0/out.html"
    assert record.doc_type == "guide"
    assert (artifact_root / "render_build.log").exists()
    log = (artifact_root / "render_build.log").read_text()
    assert "exit: 0" in log


def test_render_fails_on_nonzero_exit(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    builder = SparkRenderedBuilder(
        _rendered_config(
            _build_config(command=("{python}", "-c", "raise SystemExit(3)")),
        ),
        source_root,
        tmp_path / "artifact",
    )
    with pytest.raises(RuntimeError, match="exited 3"):
        builder.render()
    log = (tmp_path / "artifact" / "render_build.log").read_text()
    assert "exit: 3" in log


def test_render_fails_when_output_root_missing(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    builder = SparkRenderedBuilder(
        _rendered_config(
            _build_config(command=("{python}", "-c", "pass")),
        ),
        source_root,
        tmp_path / "artifact",
    )
    with pytest.raises(RuntimeError, match="output root"):
        builder.render()


def test_render_respects_include_and_exclude(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    command = (
        "{python}",
        "-c",
        (
            "import pathlib; pathlib.Path('{output}').mkdir(parents=True, exist_ok=True); "
            "pathlib.Path('{output}/page.html').write_text('<html><body><main>content</main></body></html>'); "
            "pathlib.Path('{output}/index.html').write_text('<html><body>landing</body></html>'); "
            "pathlib.Path('{output}/assets.css').write_text('body{}')"
        ),
    )
    builder = SparkRenderedBuilder(
        _rendered_config(
            _build_config(command=command, exclude=("index.html",)),
        ),
        source_root,
        tmp_path / "artifact",
    )
    manifest = builder.render()
    assert {f.relative_path for f in manifest.files} == {"page.html"}


def test_manifest_is_deterministic(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    builder_a = SparkRenderedBuilder(
        _rendered_config(_build_config()),
        source_root,
        tmp_path / "artifact-a",
    )
    builder_b = SparkRenderedBuilder(
        _rendered_config(_build_config()),
        source_root,
        tmp_path / "artifact-b",
    )
    manifest_a = builder_a.render()
    manifest_b = builder_b.render()
    assert manifest_a.manifest_hash == manifest_b.manifest_hash
    serializable_a = [(f.build, f.relative_path, f.doc_type, f.language, f.canonical_url) for f in manifest_a.files]
    serializable_b = [(f.build, f.relative_path, f.doc_type, f.language, f.canonical_url) for f in manifest_b.files]
    assert serializable_a == serializable_b


def test_build_env_contains_configured_keys(tmp_path) -> None:
    source_root = _make_source_root(tmp_path)
    builder = SparkRenderedBuilder(
        _rendered_config(
            _build_config(env=(("SKIP_API", "1"), ("FAKE_ROOT", "{root}"))),
        ),
        source_root,
        tmp_path / "artifact",
    )
    env = builder._build_env(_build_config(env=(("SKIP_API", "1"), ("FAKE_ROOT", "{root}"))), source_root)
    assert env["SKIP_API"] == "1"
    assert env["FAKE_ROOT"] == str(source_root)
