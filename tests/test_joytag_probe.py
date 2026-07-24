from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.joytag_probe import (
    IMAGE_SIZE,
    ImageResult,
    TagScore,
    average_focus_scores,
    collect_images,
    normalize_focus,
    prepare_image,
    sigmoid,
    validate_focus,
)


def test_collect_images_sorts_deduplicates_and_limits(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.JPG").write_bytes(b"image")
    (tmp_path / "a.png").write_bytes(b"image")
    (tmp_path / "ignored.txt").write_text("not an image", encoding="utf-8")
    (nested / "c.webp").write_bytes(b"image")

    flat = collect_images([tmp_path], recursive=False, limit=None)
    assert [path.name for path in flat] == ["a.png", "b.JPG"]

    recursive = collect_images(
        [tmp_path, tmp_path / "a.png"],
        recursive=True,
        limit=None,
    )
    assert [path.name for path in recursive] == ["a.png", "b.JPG", "c.webp"]

    limited = collect_images([tmp_path], recursive=True, limit=2)
    assert [path.name for path in limited] == ["a.png", "b.JPG"]


def test_prepare_image_applies_square_padding_and_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "portrait.png"
    Image.new("RGB", (40, 80), (255, 0, 0)).save(source)

    prepared = prepare_image(source)

    assert prepared.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous
    # White horizontal padding surrounds the centered red portrait.
    assert (
        prepared[1, IMAGE_SIZE // 2, 0]
        > prepared[1, IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    )
    assert (
        prepared[2, IMAGE_SIZE // 2, 0]
        > prepared[2, IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    )


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    values = np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32)
    actual = sigmoid(values)

    assert np.all(np.isfinite(actual))
    assert actual[0] == pytest.approx(0)
    assert actual[1] == pytest.approx(0.5)
    assert actual[2] == pytest.approx(1)


def test_focus_values_are_normalized_and_validated() -> None:
    focus = normalize_focus(["Mating Press, POV", "pov", "all fours"])
    assert focus == ["mating_press", "pov", "all_fours"]

    validate_focus(focus, ["mating_press", "pov", "all_fours"])

    with pytest.raises(ValueError, match="mating_pres.*perhaps mating_press"):
        validate_focus(["mating_pres"], ["mating_press", "pov"])


def test_average_focus_scores_ignores_failed_images() -> None:
    results = [
        ImageResult(
            path="one.jpg",
            selected=[],
            top=[],
            focus=[TagScore("pov", 0.8), TagScore("mating_press", 0.4)],
        ),
        ImageResult(
            path="two.jpg",
            selected=[],
            top=[],
            focus=[TagScore("pov", 0.4), TagScore("mating_press", 0.2)],
        ),
        ImageResult(
            path="broken.jpg",
            selected=[],
            top=[],
            focus=[],
            error="broken",
        ),
    ]

    assert average_focus_scores(results) == [
        TagScore("pov", pytest.approx(0.6)),
        TagScore("mating_press", pytest.approx(0.3)),
    ]
