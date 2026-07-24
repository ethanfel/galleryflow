#!/usr/bin/env python3
"""Run JoyTag against local images without changing GalleryFlow.

Examples:

    python tools/joytag_probe.py /library/reference-finder/boobjob

    python tools/joytag_probe.py /library/reference-finder/boobjob \
        --focus "fellatio,deepthroat,oral,pov,kneeling" \
        --html /tmp/joytag-report.html

The pinned JoyTag ONNX model is downloaded on first use. By default it is
stored under ``data/models/joytag`` in this checkout, which is git-ignored.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import html
import io
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


MODEL_REVISION = "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
MODEL_BASE_URL = (
    "https://huggingface.co/fancyfeast/joytag/resolve/"
    f"{MODEL_REVISION}"
)
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "top_tags.txt"
MODEL_SHA256 = "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097"
TAGS_SHA256 = "32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0"

IMAGE_SIZE = 448
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
NORMALIZE_MEAN = np.asarray(
    (0.48145466, 0.4578275, 0.40821073), dtype=np.float32
).reshape(1, 1, 3)
NORMALIZE_STD = np.asarray(
    (0.26862954, 0.26130258, 0.27577711), dtype=np.float32
).reshape(1, 1, 3)


@dataclass(frozen=True)
class TagScore:
    tag: str
    score: float


@dataclass
class ImageResult:
    path: str
    selected: list[TagScore]
    top: list[TagScore]
    focus: list[TagScore]
    error: str | None = None


def default_model_dir() -> Path:
    configured = os.environ.get("JOYTAG_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "models" / "joytag"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_download(
    destination: Path,
    url: str,
    expected_sha256: str,
) -> Path:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.part"
    )
    print(f"Downloading {destination.name} from {url}", file=sys.stderr)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "GalleryFlow-JoyTag-Probe/1.0"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    last_update = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else 0
            with temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 1:
                        if total:
                            percent = downloaded * 100 / total
                            message = (
                                f"\r  {downloaded / 1024**2:.1f} MiB "
                                f"of {total / 1024**2:.1f} MiB "
                                f"({percent:.0f}%)"
                            )
                        else:
                            message = f"\r  {downloaded / 1024**2:.1f} MiB"
                        print(message, end="", file=sys.stderr, flush=True)
                        last_update = now
        if downloaded:
            print(file=sys.stderr)

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, received {actual_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def ensure_model(model_dir: Path) -> tuple[Path, Path]:
    model_path = ensure_download(
        model_dir / MODEL_FILENAME,
        f"{MODEL_BASE_URL}/{MODEL_FILENAME}",
        MODEL_SHA256,
    )
    tags_path = ensure_download(
        model_dir / TAGS_FILENAME,
        f"{MODEL_BASE_URL}/{TAGS_FILENAME}",
        TAGS_SHA256,
    )
    return model_path, tags_path


def load_tags(path: Path) -> list[str]:
    tags = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tags:
        raise RuntimeError(f"No tags found in {path}")
    return tags


def collect_images(
    inputs: Sequence[Path],
    *,
    recursive: bool,
    limit: int | None,
) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()

    for supplied in inputs:
        path = supplied.expanduser()
        if path.is_file():
            candidates: Iterable[Path] = (path,)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                (
                    candidate
                    for candidate in iterator
                    if candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda candidate: str(candidate).casefold(),
            )
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")

        for candidate in candidates:
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append(resolved)
            if limit is not None and len(images) >= limit:
                return images

    return images


def prepare_image(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        opened.seek(0)
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        padded = Image.new("RGB", (side, side), (255, 255, 255))
        padded.paste(image, (left, top))
        resized = padded.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0

    normalized = (pixels - NORMALIZE_MEAN) / NORMALIZE_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def normalize_focus(raw_values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for value in raw_value.split(","):
            tag = value.strip().lower().replace(" ", "_")
            if tag and tag not in seen:
                seen.add(tag)
                normalized.append(tag)
    return normalized


def validate_focus(focus_tags: Sequence[str], all_tags: Sequence[str]) -> None:
    known = set(all_tags)
    unknown = [tag for tag in focus_tags if tag not in known]
    if not unknown:
        return

    messages = []
    for tag in unknown:
        alternatives = difflib.get_close_matches(tag, all_tags, n=3, cutoff=0.55)
        suffix = f" (perhaps {', '.join(alternatives)})" if alternatives else ""
        messages.append(f"{tag}{suffix}")
    raise ValueError("Unknown JoyTag focus tag(s): " + "; ".join(messages))


def create_session(model_path: Path, provider: str, device_id: int):
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider was requested but is unavailable. "
            f"Available providers: {', '.join(sorted(available))}"
        )

    use_cuda = provider in {"auto", "cuda"} and "CUDAExecutionProvider" in available
    providers: list[object]
    if use_cuda:
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception:
        if provider != "auto" or not use_cuda:
            raise
        print(
            "CUDA session creation failed; retrying with CPUExecutionProvider.",
            file=sys.stderr,
        )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Unexpected JoyTag ONNX interface: "
            f"{len(inputs)} input(s), {len(outputs)} output(s)"
        )
    return session, inputs[0].name, outputs[0].name


def ranked_scores(
    probabilities: np.ndarray,
    tags: Sequence[str],
    indexes: Iterable[int],
) -> list[TagScore]:
    return [
        TagScore(tag=tags[index], score=float(probabilities[index]))
        for index in indexes
    ]


def run_probe(
    image_paths: Sequence[Path],
    *,
    model_path: Path,
    tags: Sequence[str],
    focus_tags: Sequence[str],
    provider: str,
    device_id: int,
    batch_size: int,
    threshold: float,
    top_count: int,
) -> tuple[list[ImageResult], list[TagScore], str]:
    session, input_name, output_name = create_session(
        model_path, provider, device_id
    )
    active_provider = session.get_providers()[0]
    tag_indexes = {tag: index for index, tag in enumerate(tags)}
    focus_indexes = [tag_indexes[tag] for tag in focus_tags]
    results_by_path: dict[Path, ImageResult] = {}
    probability_total = np.zeros(len(tags), dtype=np.float64)
    successful = 0

    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        prepared: list[np.ndarray] = []
        valid_paths: list[Path] = []

        for path in paths:
            try:
                prepared.append(prepare_image(path))
                valid_paths.append(path)
            except Exception as exc:
                results_by_path[path] = ImageResult(
                    path=str(path),
                    selected=[],
                    top=[],
                    focus=[],
                    error=f"{type(exc).__name__}: {exc}",
                )

        if not prepared:
            continue

        batch = np.stack(prepared).astype(np.float32, copy=False)
        logits = np.asarray(
            session.run([output_name], {input_name: batch})[0],
            dtype=np.float32,
        )
        if logits.shape != (len(valid_paths), len(tags)):
            raise RuntimeError(
                "Unexpected JoyTag output shape: "
                f"received {logits.shape}, expected "
                f"({len(valid_paths)}, {len(tags)})"
            )
        probabilities = sigmoid(logits)

        for path, scores in zip(valid_paths, probabilities, strict=True):
            ordered = np.argsort(scores)[::-1]
            selected_indexes = [
                int(index)
                for index in ordered
                if float(scores[index]) > threshold
            ][:top_count]
            top_indexes = [int(index) for index in ordered[:top_count]]
            results_by_path[path] = ImageResult(
                path=str(path),
                selected=ranked_scores(scores, tags, selected_indexes),
                top=ranked_scores(scores, tags, top_indexes),
                focus=ranked_scores(scores, tags, focus_indexes),
            )
            probability_total += scores.astype(np.float64)
            successful += 1

        completed = min(start + len(paths), len(image_paths))
        print(
            f"\rTagged {completed}/{len(image_paths)} image(s)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    print(file=sys.stderr)
    results = [results_by_path[path] for path in image_paths]
    if successful:
        average = probability_total / successful
        order = np.argsort(average)[::-1][:top_count]
        summary = ranked_scores(average, tags, (int(index) for index in order))
    else:
        summary = []
    return results, summary, active_provider


def terminal_report(
    results: Sequence[ImageResult],
    summary: Sequence[TagScore],
    focus_summary: Sequence[TagScore],
    *,
    threshold: float,
    provider: str,
) -> None:
    print(f"Provider: {provider}")
    print(f"Threshold: {threshold:.2f}")
    print()

    for result in results:
        print(result.path)
        if result.error:
            print(f"  ERROR: {result.error}")
            print()
            continue

        selected = ", ".join(
            f"{item.tag}={item.score:.3f}" for item in result.selected
        )
        print(f"  selected: {selected or '(none)'}")
        top = ", ".join(f"{item.tag}={item.score:.3f}" for item in result.top)
        print(f"  top:      {top}")
        if result.focus:
            focus = ", ".join(
                f"{item.tag}={item.score:.3f}" for item in result.focus
            )
            print(f"  focus:    {focus}")
        print()

    if summary:
        average = ", ".join(
            f"{item.tag}={item.score:.3f}" for item in summary
        )
        print(f"Folder average: {average}")
    if focus_summary:
        average = ", ".join(
            f"{item.tag}={item.score:.3f}" for item in focus_summary
        )
        print(f"Folder focus average: {average}")


def average_focus_scores(results: Sequence[ImageResult]) -> list[TagScore]:
    usable = [result for result in results if not result.error and result.focus]
    if not usable:
        return []

    totals = {item.tag: 0.0 for item in usable[0].focus}
    for result in usable:
        for item in result.focus:
            totals[item.tag] += item.score
    return [
        TagScore(tag=tag, score=total / len(usable))
        for tag, total in totals.items()
    ]


def thumbnail_data_url(path: Path, max_size: int = 300) -> str:
    with Image.open(path) as opened:
        opened.seek(0)
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def score_chips(scores: Sequence[TagScore], threshold: float) -> str:
    chips = []
    for item in scores:
        active = " active" if item.score > threshold else ""
        chips.append(
            f'<span class="chip{active}">'
            f"{html.escape(item.tag)} "
            f"<strong>{item.score:.3f}</strong></span>"
        )
    return "".join(chips)


def write_html_report(
    destination: Path,
    results: Sequence[ImageResult],
    summary: Sequence[TagScore],
    focus_summary: Sequence[TagScore],
    *,
    threshold: float,
    provider: str,
) -> None:
    cards = []
    for result in results:
        escaped_path = html.escape(result.path)
        if result.error:
            cards.append(
                '<article class="card error">'
                f"<h2>{escaped_path}</h2>"
                f"<p>{html.escape(result.error)}</p>"
                "</article>"
            )
            continue

        try:
            image_url = thumbnail_data_url(Path(result.path))
            image_markup = (
                f'<img src="{image_url}" alt="{escaped_path}">'
            )
        except Exception as exc:
            image_markup = (
                '<div class="thumb-error">'
                f"Thumbnail error: {html.escape(str(exc))}</div>"
            )

        focus_markup = ""
        if result.focus:
            focus_markup = (
                "<h3>Focus tags</h3>"
                f'<div class="chips focus">'
                f"{score_chips(result.focus, threshold)}</div>"
            )

        cards.append(
            '<article class="card">'
            f"{image_markup}"
            f"<h2>{escaped_path}</h2>"
            "<h3>Selected tags</h3>"
            f'<div class="chips selected">'
            f"{score_chips(result.selected, threshold) or '<em>None</em>'}</div>"
            f"{focus_markup}"
            "<details><summary>Top scores regardless of threshold</summary>"
            f'<div class="chips">{score_chips(result.top, threshold)}</div>'
            "</details>"
            "</article>"
        )

    summary_markup = score_chips(summary, threshold)
    focus_summary_markup = ""
    if focus_summary:
        focus_summary_markup = (
            "<h3>Requested focus-tag averages</h3>"
            f'<div class="chips focus">'
            f"{score_chips(focus_summary, threshold)}</div>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JoyTag probe</title>
<style>
:root {{
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #0d0d12;
  color: #f4f2f7;
}}
body {{ margin: 0; padding: 28px; }}
header {{ max-width: 1200px; margin: 0 auto 24px; }}
h1 {{ margin: 0 0 8px; }}
.meta {{ color: #aaa5b2; }}
.summary {{
  margin-top: 18px; padding: 16px; border: 1px solid #302b38;
  border-radius: 14px; background: #17151d;
}}
.grid {{
  max-width: 1600px; margin: auto; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;
}}
.card {{
  min-width: 0; padding: 16px; border: 1px solid #302b38;
  border-radius: 16px; background: #17151d;
}}
.card img {{
  display: block; width: 100%; height: 320px; object-fit: contain;
  border-radius: 12px; background: #0b0a0e;
}}
.card h2 {{
  margin: 14px 0; font-size: 13px; color: #c9c4cf;
  overflow-wrap: anywhere;
}}
.card h3 {{ margin: 14px 0 7px; font-size: 12px; color: #96909e; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{
  padding: 5px 8px; border-radius: 999px; background: #25212d;
  color: #aaa5b2; font-size: 12px;
}}
.chip.active {{ background: #52305f; color: #fff; }}
.focus .chip {{ outline: 1px solid #61536d; }}
details {{ margin-top: 14px; }}
summary {{ cursor: pointer; color: #aaa5b2; margin-bottom: 8px; }}
.error {{ border-color: #7a3346; }}
.thumb-error {{ padding: 40px 12px; color: #d98a9f; }}
</style>
</head>
<body>
<header>
  <h1>JoyTag probe</h1>
  <div class="meta">{len(results)} image(s) · provider {html.escape(provider)}
    · selected threshold {threshold:.2f}</div>
  <section class="summary">
    <h3>Average scores across successful images</h3>
    <div class="chips">{summary_markup}</div>
    {focus_summary_markup}
  </section>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def write_json_report(
    destination: Path,
    results: Sequence[ImageResult],
    summary: Sequence[TagScore],
    focus_summary: Sequence[TagScore],
    *,
    threshold: float,
    provider: str,
) -> None:
    payload = {
        "model": f"fancyfeast/joytag@{MODEL_REVISION}",
        "provider": provider,
        "threshold": threshold,
        "summary": [asdict(item) for item in summary],
        "focus_summary": [asdict(item) for item in focus_summary],
        "images": [
            {
                **asdict(result),
                "selected": [asdict(item) for item in result.selected],
                "top": [asdict(item) for item in result.top],
                "focus": [asdict(item) for item in result.focus],
            }
            for result in results
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned JoyTag ONNX model on an image or folder. "
            "This is standalone and does not read or modify GalleryFlow's DB."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Image file(s) or folder(s) to inspect.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively.",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="Stop after this many images.",
    )
    parser.add_argument(
        "--threshold",
        type=probability,
        default=0.4,
        help="Selection threshold (official default: 0.4).",
    )
    parser.add_argument(
        "--top",
        type=positive_integer,
        default=24,
        help="Maximum selected/top scores to show per image (default: 24).",
    )
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        metavar="TAG[,TAG...]",
        help=(
            "Always display scores for these tags, even below threshold. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=8,
        help="ONNX inference batch size (default: 8).",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution provider (default: auto).",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir(),
        help=(
            "JoyTag model cache directory. Defaults to JOYTAG_MODEL_DIR or "
            "data/models/joytag."
        ),
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Write a self-contained visual HTML report.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write the displayed scores as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        images = collect_images(
            args.inputs,
            recursive=args.recursive,
            limit=args.limit,
        )
        if not images:
            raise RuntimeError("No supported images found in the supplied path(s)")

        model_path, tags_path = ensure_model(args.model_dir)
        tags = load_tags(tags_path)
        focus_tags = normalize_focus(args.focus)
        validate_focus(focus_tags, tags)

        print(
            f"Running JoyTag on {len(images)} image(s) "
            f"with batch size {args.batch_size}...",
            file=sys.stderr,
        )
        results, summary, provider = run_probe(
            images,
            model_path=model_path,
            tags=tags,
            focus_tags=focus_tags,
            provider=args.provider,
            device_id=args.device_id,
            batch_size=args.batch_size,
            threshold=args.threshold,
            top_count=args.top,
        )
        focus_summary = average_focus_scores(results)
        terminal_report(
            results,
            summary,
            focus_summary,
            threshold=args.threshold,
            provider=provider,
        )

        if args.html:
            destination = args.html.expanduser().resolve()
            write_html_report(
                destination,
                results,
                summary,
                focus_summary,
                threshold=args.threshold,
                provider=provider,
            )
            print(f"HTML report: {destination}", file=sys.stderr)
        if args.json:
            destination = args.json.expanduser().resolve()
            write_json_report(
                destination,
                results,
                summary,
                focus_summary,
                threshold=args.threshold,
                provider=provider,
            )
            print(f"JSON report: {destination}", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
