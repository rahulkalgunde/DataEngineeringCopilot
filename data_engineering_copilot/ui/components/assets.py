"""Bundled Lottie animation specs (inline dicts, fully offline-safe).

Each entry is a minimal-but-valid Lottie JSON structure consumed by lottie-web
inside the self-built player in ``animations.render_lottie_badge``. Keeping
them as data means no network request is ever needed and the package stays
self-contained. All specs are validated on import and any failure is handled
by the CSS/SVG fallback layer at render time.
"""

from __future__ import annotations

LOTTIE_VERSION = "5.7.4"
FRAME_RATE = 30
DURATION_FRAMES = 60  # 2 seconds at 30 fps


def _static(value: float | list[float]) -> dict:
    return {"a": 0, "k": value}


def _animated(frames: list[tuple[int, list[float] | float]]) -> dict:
    """Animated keyframe property.

    ``frames`` is a list of ``(frame, value)`` pairs; easing is applied between
    consecutive keys so small hand-built specs animate smoothly.
    """
    keys: list[dict] = []
    for idx, (t, value) in enumerate(frames):
        key: dict = {"t": t, "s": list(value) if isinstance(value, (list, tuple)) else value}
        if idx > 0:
            key["i"] = {"x": [0.667], "y": [1.0]}
            key["o"] = {"x": [0.333], "y": [0.0]}
        keys.append(key)
    return {"a": 1, "k": keys}


def _transform(position: dict | None = None, rotation: dict | None = None, scale: dict | None = None) -> dict:
    """Composite layer transform block."""
    return {
        "o": _static(100),
        "r": rotation or _static(0.0),
        "p": position or _static([0.0, 0.0]),
        "a": _static([0.0, 0.0, 0.0]),
        "s": scale or _static([100.0, 100.0, 100.0]),
        "sk": {"a": 0, "k": 0.0},
        "sa": {"a": 0, "k": 0.0},
    }


def _color(hex_color: str, alpha: float = 1.0) -> dict:
    h = hex_color.lstrip("#")
    return {"a": 0, "k": [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, alpha]}


def _shape_layer(layer_id: int, name: str, shapes: list[dict], out_point: int = DURATION_FRAMES) -> dict:
    return {
        "ddd": 0,
        "ind": layer_id,
        "ty": 4,
        "nm": name,
        "sr": 1,
        "ks": _transform(),
        "ao": 0,
        "shapes": shapes,
        "ip": 0,
        "op": out_point,
        "st": 0,
        "bm": 0,
    }


def _group(name: str, items: list[dict]) -> dict:
    return {"ty": "gr", "nm": name, "it": items}


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> dict:
    return {"ty": "el", "d": 1, "s": _static([rx * 2, ry * 2]), "p": _static([cx, cy]), "nm": "el"}


def _rect(x: float, y: float, w: float, h: float, radius: float = 0.0) -> dict:
    return {"ty": "rc", "d": 1, "s": _static([w, h]), "p": _static([x, y]), "r": _static(radius), "nm": "rect"}


def _fill(color: str) -> dict:
    return {"ty": "fl", "c": _color(color), "o": _static(100), "r": 1, "nm": "fill", "bm": 0}


def _stroke(color: str, width: float) -> dict:
    return {
        "ty": "st",
        "c": _color(color),
        "o": _static(100),
        "w": _static(width),
        "lc": 2,
        "lj": 2,
        "nm": "stroke",
        "bm": 0,
    }


def _transform_item(
    dx: float,
    dy: float,
    *,
    position: dict | None = None,
    rotation: dict | None = None,
    scale: dict | None = None,
) -> dict:
    return {
        "ty": "tr",
        "p": position or _static([dx, dy]),
        "a": _static([0.0, 0.0]),
        "s": scale or _static([100.0, 100.0]),
        "r": rotation or _static(0.0),
        "o": _static(100),
        "sk": _static(0.0),
        "sa": _static(0.0),
    }


def _build_spec(layers: list[dict]) -> dict:
    return {
        "v": LOTTIE_VERSION,
        "fr": FRAME_RATE,
        "ip": 0,
        "op": DURATION_FRAMES,
        "w": 120,
        "h": 120,
        "nm": "data-engineering-copilot",
        "ddd": 0,
        "assets": [],
        "layers": layers,
    }


def _scan_doc() -> dict:
    """Document parsing: rounded-paper outline + traveling scan line."""
    paper = _group("doc", [_rect(60.0, 62.0, 80.0, 92.0, radius=6.0), _stroke("#94A3B8", 3.0), _transform_item(0, 0)])
    travel = _animated([(0, [60.0, 22.0]), (30, [60.0, 96.0]), (60, [60.0, 22.0])])
    scanner = _group(
        "scanner", [_rect(60.0, 0.0, 56.0, 4.0, radius=2.0), _fill("#3B82F6"), _transform_item(0, 0, position=travel)]
    )
    return _build_spec([_shape_layer(1, "document", [paper]), _shape_layer(2, "scanner", [scanner])])


def _neural_grid() -> dict:
    """Vector embedding: a pulsing dot grid (neural net abstract)."""
    points = [(24, 30), (56, 22), (88, 30), (32, 66), (64, 58), (96, 66), (60, 92), (24, 92)]
    layers: list[dict] = []
    for idx, (cx, cy) in enumerate(points):
        pulse = _animated([(0, [100.0, 100.0]), (30, [150.0, 150.0]), (60, [100.0, 100.0])])
        layers.append(
            _shape_layer(
                idx + 1,
                f"node-{idx}",
                [
                    _group(
                        f"dot-{idx}", [_ellipse(cx, cy, 6.0, 6.0), _fill("#6366F1"), _transform_item(0, 0, scale=pulse)]
                    )
                ],
            )
        )
    return _build_spec(layers)


def _radar_search() -> dict:
    """Qdrant vector search: radar sweep + orbiting candidate point."""
    ring = _group("ring", [_ellipse(60.0, 60.0, 48.0, 48.0), _stroke("#38BDF8", 3.0), _transform_item(0, 0)])
    spin = _animated([(0, 0.0), (60, 360.0)])
    sweeper = _group(
        "sweeper", [_rect(60.0, 0.0, 34.0, 3.0, radius=1.5), _fill("#F59E0B"), _transform_item(0, 0, rotation=spin)]
    )
    orbit = _animated(
        [(0, [60.0, 20.0]), (15, [88.0, 40.0]), (30, [60.0, 100.0]), (45, [32.0, 40.0]), (60, [60.0, 20.0])]
    )
    hit = _group("orbiter", [_ellipse(0.0, 0.0, 5.0, 5.0), _fill("#22D3EE"), _transform_item(0, 0, position=orbit)])
    return _build_spec(
        [_shape_layer(1, "radar", [ring]), _shape_layer(2, "sweep", [sweeper]), _shape_layer(3, "hit", [hit])]
    )


def _typing_dots() -> dict:
    """LLM answer streaming: three staggered bouncing dots."""
    layers: list[dict] = []
    for idx, cx in enumerate((30, 60, 90)):
        stagger = idx * 8
        bounce = _animated(
            [
                (0 + stagger, [100.0, 100.0]),
                (15 + stagger, [45.0, 45.0]),
                (30 + stagger, [100.0, 100.0]),
                (60, [100.0, 100.0]),
            ]
        )
        layers.append(
            _shape_layer(
                idx + 1,
                f"dot-{idx}",
                [
                    _group(
                        f"dots-{idx}",
                        [_ellipse(cx, 76.0, 10.0, 10.0), _fill("#10B981"), _transform_item(0, 0, scale=bounce)],
                    )
                ],
            )
        )
    return _build_spec(layers)


ANIMATIONS: dict[str, dict] = {
    "parse": _scan_doc(),
    "embed": _neural_grid(),
    "search": _radar_search(),
    "generate": _typing_dots(),
}


def validate_animations() -> list[str]:
    """Return names of malformed specs (empty when all are valid)."""
    problems: list[str] = []
    for name, spec in ANIMATIONS.items():
        if (
            any(key not in spec for key in ("v", "fr", "ip", "op", "w", "h", "layers"))
            or not isinstance(spec["layers"], list)
            or not spec["layers"]
        ):
            problems.append(name)
        else:
            for layer in spec["layers"]:
                if not isinstance(layer, dict) or "shapes" not in layer or "ks" not in layer:
                    problems.append(name)
                    break
    return problems


_VALIDATION_ERRORS = validate_animations()
if _VALIDATION_ERRORS:
    raise RuntimeError(f"Invalid bundled Lottie specs: {_VALIDATION_ERRORS}")
