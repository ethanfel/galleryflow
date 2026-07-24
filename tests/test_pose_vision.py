from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.pose_vision import (
    RTMO_L_ARCHIVE_MEMBER,
    RTMO_L_ARCHIVE_SHA256,
    RTMO_L_ARCHIVE_URL,
    RTMO_L_MODEL_KEY,
    RTMO_L_MODEL_SHA256,
    PoseFrame,
    PoseInferenceError,
    PoseInvalidImageError,
    PoseModelIntegrityError,
    PoseModelPreparationError,
    RTMOPoseEstimator,
    mirror_pose_frame,
    pose_geometry_match,
    scene_kind,
    skeleton_data_uri,
    skeleton_svg,
)


def encoded_image(size: tuple[int, int] = (100, 200)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (240, 30, 10)).save(output, format="PNG")
    return output.getvalue()


def model_zip(payload: bytes, *, symlink: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if symlink:
            info = zipfile.ZipInfo(RTMO_L_ARCHIVE_MEMBER)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"somewhere.onnx")
        else:
            archive.writestr(RTMO_L_ARCHIVE_MEMBER, payload)
            archive.writestr("deploy.json", b"{}")
    return output.getvalue()


class FakeSession:
    def __init__(
        self,
        dets: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        *,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.dets = (
            dets
            if dets is not None
            else np.empty((1, 0, 5), dtype=np.float32)
        )
        self.keypoints = (
            keypoints
            if keypoints is not None
            else np.empty((1, 0, 17, 3), dtype=np.float32)
        )
        self.provider = provider
        self.feeds: list[np.ndarray] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input")]

    def get_providers(self) -> list[str]:
        return [self.provider]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        self.feeds.append(input_feed["input"].copy())
        return [self.dets, self.keypoints]


class DynamicBatchSession(FakeSession):
    """Mimic the pinned model's batch-dynamic input and output axes."""

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        batch = np.asarray(input_feed["input"])
        self.feeds.append(batch.copy())
        batch_size = batch.shape[0]
        dets = np.tile(
            np.asarray([20, 20, 230, 220, 0.9], dtype=np.float32),
            (batch_size, 1, 1),
        )
        keypoints = np.zeros(
            (batch_size, 1, 17, 3), dtype=np.float32
        )
        for joint in range(17):
            keypoints[:, 0, joint, :] = (
                50 + joint * 8,
                40 + joint * 8,
                0.9,
            )
        return [dets, keypoints]


def test_official_rtmo_l_artifact_is_fully_pinned() -> None:
    assert RTMO_L_ARCHIVE_URL.endswith(
        "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
    )
    assert RTMO_L_ARCHIVE_SHA256 == (
        "17b361174d759d974879f9fb46d564ae658d004bfa070e6f1c9ad275d3fd6b87"
    )
    assert RTMO_L_MODEL_SHA256 == (
        "090096ca90f29163cc4f67137dcc0cd4b2ee95ea0af11764fbfda88dd2ae1140"
    )
    assert "pose-geometry-v1" in RTMO_L_MODEL_KEY


@pytest.mark.asyncio
async def test_prepare_verifies_zip_and_member_then_publishes_atomically(
    tmp_path: Path,
) -> None:
    model = b"a deterministic tiny stand-in for end2end.onnx"
    archive = model_zip(model)
    downloads: list[tuple[str, Path]] = []

    async def downloader(url: str, destination: Path) -> None:
        downloads.append((url, destination))
        assert not (tmp_path / "models" / "rtmo-l.onnx").exists()
        destination.write_bytes(archive)

    path = tmp_path / "models" / "rtmo-l.onnx"
    estimator = RTMOPoseEstimator(
        path,
        model_url="https://models.example/rtmo-l.zip",
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        model_sha256=hashlib.sha256(model).hexdigest(),
        downloader=downloader,
    )

    assert await estimator.prepare() == path
    assert path.read_bytes() == model
    assert len(downloads) == 1
    assert list(path.parent.glob("*.part")) == []
    assert await estimator.prepare() == path
    assert len(downloads) == 1


@pytest.mark.asyncio
async def test_prepare_never_replaces_existing_file_with_wrong_member(
    tmp_path: Path,
) -> None:
    original = b"existing model is kept until a replacement is verified"
    path = tmp_path / "rtmo-l.onnx"
    path.write_bytes(original)
    archive = model_zip(b"wrong model")

    async def downloader(_: str, destination: Path) -> None:
        destination.write_bytes(archive)

    estimator = RTMOPoseEstimator(
        path,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        model_sha256=hashlib.sha256(b"expected model").hexdigest(),
        downloader=downloader,
    )
    with pytest.raises(PoseModelIntegrityError, match="model failed SHA-256"):
        await estimator.prepare()
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.asyncio
async def test_prepare_rejects_a_symlink_model_member(tmp_path: Path) -> None:
    archive = model_zip(b"ignored", symlink=True)

    async def downloader(_: str, destination: Path) -> None:
        destination.write_bytes(archive)

    estimator = RTMOPoseEstimator(
        tmp_path / "rtmo.onnx",
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        model_sha256=hashlib.sha256(b"somewhere.onnx").hexdigest(),
        downloader=downloader,
    )
    with pytest.raises(PoseModelPreparationError, match="regular file"):
        await estimator.prepare()


def test_auto_provider_prefers_cuda_and_exposes_status(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    calls: list[tuple[list[str], int]] = []
    session = FakeSession(provider="CUDAExecutionProvider")

    def factory(_: Path, providers: list[str], threads: int) -> FakeSession:
        calls.append((providers, threads))
        return session

    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
        session_threads=99,
    )
    estimator.infer_bytes(encoded_image())

    assert calls == [(["CUDAExecutionProvider", "CPUExecutionProvider"], 4)]
    assert estimator.provider_status() == {
        "requested": "auto",
        "active": "CUDAExecutionProvider",
        "available": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "fallback": False,
        "message": "",
    }


def test_auto_cuda_session_failure_falls_back_to_cpu(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    calls: list[list[str]] = []

    def factory(_: Path, providers: list[str], __: int) -> FakeSession:
        calls.append(providers)
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("driver mismatch")
        return FakeSession(provider="CPUExecutionProvider")

    estimator = RTMOPoseEstimator(
        path,
        execution_provider="auto",
        available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
    )
    estimator.infer_bytes(encoded_image())

    assert calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    status = estimator.provider_status()
    assert status["active"] == "CPUExecutionProvider"
    assert status["fallback"] is True
    assert "initialization failed" in status["message"]


def test_explicit_cuda_never_silently_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    unavailable = RTMOPoseEstimator(
        path,
        execution_provider="cuda",
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: FakeSession(),
    )
    with pytest.raises(PoseModelPreparationError, match="CUDAExecutionProvider"):
        unavailable.infer_bytes(encoded_image())

    def broken_cuda(*_: Any) -> FakeSession:
        raise RuntimeError("driver mismatch")

    broken = RTMOPoseEstimator(
        path,
        execution_provider="cuda",
        available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=broken_cuda,
    )
    with pytest.raises(PoseInferenceError, match="create"):
        broken.infer_bytes(encoded_image())


def test_cpu_only_runtime_is_a_reported_auto_fallback(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: FakeSession(),
    )
    estimator.infer_bytes(encoded_image())
    status = estimator.provider_status()
    assert status["active"] == "CPUExecutionProvider"
    assert status["fallback"] is True
    assert "unavailable" in status["message"]


def _model_point(x: float, y: float) -> tuple[float, float]:
    # A 100x200 portrait becomes 320x640 at top-left, with right-side padding.
    return x * 320, y * 640


def test_inference_maps_back_to_original_and_filters_unreliable_people(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    dets = np.asarray(
        [
            [0, 0, 150, 640, 0.90],
            [0, 0, 150, 640, 0.20],  # detector score is too low
            [0, 0, 150, 640, 0.80],  # too few visible joints
            [170, 0, 320, 640, 0.70],  # partly occluded but reliable
        ],
        dtype=np.float32,
    )[None]
    keypoints = np.zeros((1, 4, 17, 3), dtype=np.float32)
    for person in range(4):
        for joint in range(17):
            keypoints[0, person, joint, :2] = _model_point(
                0.15 + joint * 0.04, 0.10 + joint * 0.045
            )
    keypoints[0, 0, :, 2] = 0.9
    keypoints[0, 1, :, 2] = 0.9
    keypoints[0, 2, :2, 2] = 0.9
    keypoints[0, 3, :4, 2] = 0.9
    # A confident padded-area prediction must become a missing observation.
    keypoints[0, 0, 16] = (500, 300, 0.99)
    session = FakeSession(dets, keypoints)
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: session,
    )

    frame = estimator.infer_bytes(encoded_image())

    assert frame.person_count == 2
    assert frame.scene_kind == "couple"
    assert frame.image_size == (100, 200)
    assert frame.provider == "CPUExecutionProvider"
    assert frame.keypoints.shape == (2, 17, 2)
    assert frame.keypoints[0, 0] == pytest.approx((0.15, 0.10), abs=2e-3)
    assert frame.confidences[0, 16] == 0
    assert np.all((frame.keypoints >= 0) & (frame.keypoints <= 1))
    assert session.feeds[0].shape == (1, 3, 640, 640)
    assert session.feeds[0].dtype == np.float32
    # Red RGB input reaches the model in OpenCV/BGR channel order.
    assert session.feeds[0][0, 2, 320, 160] > 200
    assert session.feeds[0][0, 0, 320, 160] < 30
    # Official RTMO parity: portrait padding is to the right, not centered.
    assert np.all(session.feeds[0][0, :, 320, 500] == 114)


def test_infer_many_runs_one_dynamic_batch_and_matches_single_results(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    payloads = [
        encoded_image((100, 200)),
        encoded_image((200, 100)),
        encoded_image((120, 120)),
    ]
    batch_session = DynamicBatchSession()
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: batch_session,
    )

    frames = estimator.infer_many_bytes(payloads)

    assert len(batch_session.feeds) == 1
    assert batch_session.feeds[0].shape == (3, 3, 640, 640)
    assert [frame.image_size for frame in frames] == [
        (100, 200),
        (200, 100),
        (120, 120),
    ]

    single_session = DynamicBatchSession()
    single_estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: single_session,
    )
    expected = [single_estimator.infer_bytes(payload) for payload in payloads]
    for actual, single in zip(frames, expected, strict=True):
        np.testing.assert_allclose(actual.keypoints, single.keypoints)
        np.testing.assert_allclose(actual.confidences, single.confidences)
        np.testing.assert_allclose(actual.boxes, single.boxes)
        np.testing.assert_allclose(actual.person_scores, single.person_scores)
        assert actual.image_size == single.image_size
        assert actual.provider == single.provider


def test_infer_many_validates_inputs_before_session_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    session = DynamicBatchSession()
    factory_calls = 0

    def factory(*_: Any) -> DynamicBatchSession:
        nonlocal factory_calls
        factory_calls += 1
        return session

    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=factory,
    )
    valid = encoded_image()

    assert estimator.infer_many_bytes([]) == []
    with pytest.raises(TypeError, match="sequence"):
        estimator.infer_many_bytes(valid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes-like"):
        estimator.infer_many_bytes([valid, object()])  # type: ignore[list-item]
    with pytest.raises(PoseInvalidImageError):
        estimator.infer_many_bytes([valid, b"not-an-image"])
    assert factory_calls == 0
    assert session.feeds == []


def test_infer_many_rejects_wrong_batch_output_size(tmp_path: Path) -> None:
    class ShortBatchSession(DynamicBatchSession):
        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            dets, keypoints = super().run(output_names, input_feed)
            return [dets[:-1], keypoints]

    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: ShortBatchSession(),
    )

    with pytest.raises(PoseInferenceError, match="batch output shapes"):
        estimator.infer_many_bytes([encoded_image(), encoded_image()])


def test_class_agnostic_nms_removes_duplicate_person_proposals(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    dets = np.asarray(
        [[0, 0, 320, 640, 0.9], [4, 2, 316, 638, 0.8]], dtype=np.float32
    )[None]
    keypoints = np.zeros((1, 2, 17, 3), dtype=np.float32)
    for person in range(2):
        for joint in range(17):
            keypoints[0, person, joint, :2] = _model_point(
                0.2 + joint * 0.03, 0.1 + joint * 0.045
            )
        keypoints[0, person, :, 2] = 0.9
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: FakeSession(dets, keypoints),
    )

    frame = estimator.infer_bytes(encoded_image())

    assert frame.person_count == 1
    assert frame.scene_kind == "solo"
    assert frame.person_scores[0] > 0.8


def test_bad_model_output_contract_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"injected")
    bad = FakeSession(
        np.ones((1, 2, 5), dtype=np.float32),
        np.ones((1, 2, 16, 3), dtype=np.float32),
    )
    estimator = RTMOPoseEstimator(
        path,
        available_providers=lambda: ["CPUExecutionProvider"],
        session_factory=lambda *_: bad,
    )
    with pytest.raises(PoseInferenceError, match="output shapes"):
        estimator.infer_bytes(encoded_image())


def person_pose(*, offset_x: float = 0, bend: float = 0) -> np.ndarray:
    # Deliberately asymmetric 17-joint shape so mirroring and left/right swaps
    # are both observable.
    points = np.asarray(
        [
            (0.50, 0.05), (0.46, 0.04), (0.55, 0.06), (0.41, 0.06),
            (0.61, 0.09), (0.38, 0.25), (0.62, 0.28), (0.27, 0.43),
            (0.72, 0.48), (0.18, 0.61), (0.83, 0.56), (0.43, 0.57),
            (0.58, 0.59), (0.39, 0.76), (0.64, 0.78), (0.34, 0.97),
            (0.72, 0.94),
        ],
        dtype=np.float32,
    )
    points[:, 0] += offset_x
    points[9, 1] -= bend
    points[10, 1] += bend
    return points


def frame_from_people(
    people: list[np.ndarray],
    confidences: list[np.ndarray] | None = None,
) -> PoseFrame:
    keypoints = np.stack(people).astype(np.float32)
    confidence = (
        np.stack(confidences).astype(np.float32)
        if confidences is not None
        else np.full((len(people), 17), 0.9, dtype=np.float32)
    )
    boxes = np.asarray(
        [
            (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())
            for points in keypoints
        ],
        dtype=np.float32,
    )
    return PoseFrame(
        keypoints=keypoints,
        confidences=confidence,
        boxes=boxes,
        person_scores=np.full((len(people),), 0.9, dtype=np.float32),
        image_size=(1000, 1000),
    )


def test_geometry_is_confidence_aware_and_ignores_one_missing_joint() -> None:
    reference = frame_from_people([person_pose()])
    missing = np.full((17,), 0.9, dtype=np.float32)
    missing[9] = 0.001
    candidate = frame_from_people([person_pose()], [missing])

    match = pose_geometry_match(reference, candidate)

    assert match.score > 0.95
    assert match.coverage > 0.95
    assert match.reliable is True


def test_pose_frame_cache_round_trip_is_strictly_validated() -> None:
    original = frame_from_people([person_pose()])
    restored = PoseFrame.from_dict(original.as_dict())
    assert np.array_equal(restored.keypoints, original.keypoints)
    assert np.array_equal(restored.confidences, original.confidences)
    assert restored.image_size == original.image_size

    empty = PoseFrame(
        keypoints=np.empty((0, 17, 2), dtype=np.float32),
        confidences=np.empty((0, 17), dtype=np.float32),
        boxes=np.empty((0, 4), dtype=np.float32),
        person_scores=np.empty((0,), dtype=np.float32),
        image_size=(640, 480),
    )
    restored_empty = PoseFrame.from_dict(empty.as_dict())
    assert restored_empty.person_count == 0
    assert restored_empty.keypoints.shape == (0, 17, 2)

    malformed = original.as_dict()
    malformed["keypoints"] = [[[float("nan"), 0.2]] * 17]
    with pytest.raises(ValueError, match="non-finite"):
        PoseFrame.from_dict(malformed)

    malformed = original.as_dict()
    malformed["confidences"] = [[0.9] * 16]
    with pytest.raises(ValueError, match="confidences.*shape"):
        PoseFrame.from_dict(malformed)


def test_three_joint_match_is_available_but_not_marked_reliable() -> None:
    sparse_confidence = np.zeros((17,), dtype=np.float32)
    sparse_confidence[[5, 6, 11]] = 0.9
    sparse = frame_from_people([person_pose()], [sparse_confidence])

    match = pose_geometry_match(sparse, sparse)

    assert match.score == pytest.approx(1.0, abs=1e-6)
    assert match.common_joints == 3
    assert match.reliable is False


def test_five_body_joint_match_stays_reliable_but_is_not_precise() -> None:
    sparse_confidence = np.zeros((17,), dtype=np.float32)
    sparse_confidence[[5, 6, 7, 8, 11]] = 0.9
    sparse = frame_from_people([person_pose()], [sparse_confidence])

    match = pose_geometry_match(sparse, sparse)

    assert match.reliable is True
    assert match.precise is False
    assert match.body_common_joints == 5
    assert match.core_anchor_joints == 3
    assert match.distal_limb_joints == 2
    assert match.supported_limb_segments == 2


def test_face_heavy_sparse_body_match_cannot_be_reliable() -> None:
    # Six mutually confident joints satisfy the common-joint rule, but five
    # are facial points and only one describes the body pose. This mirrors the
    # false sparse people observed in extreme gallery crops.
    sparse_confidence = np.zeros((17,), dtype=np.float32)
    sparse_confidence[[0, 1, 2, 3, 4, 5]] = 0.9
    sparse = frame_from_people([person_pose()], [sparse_confidence])

    match = pose_geometry_match(sparse, sparse)

    assert match.common_joints == 6
    assert match.mean_joint_confidence > 0.8
    assert match.minimum_body_confidence < 0.15
    assert match.reliable is False
    assert match.precise is False
    assert match.body_common_joints == 1
    assert match.core_anchor_joints == 1
    assert match.distal_limb_joints == 0
    assert match.supported_limb_segments == 0


def test_person_count_mismatch_can_be_reliable_but_not_precise() -> None:
    reference = frame_from_people([person_pose()])
    candidate = frame_from_people(
        [person_pose(), person_pose(offset_x=0.3, bend=0.08)]
    )

    match = pose_geometry_match(reference, candidate)

    assert match.reliable is True
    assert match.reference_count == 1
    assert match.candidate_count == 2
    assert match.precise is False
    assert match.as_dict()["diagnostics"]["exact_person_count"] is False


def test_well_observed_same_mirrored_and_couple_matches_are_precise() -> None:
    solo = frame_from_people([person_pose(bend=0.10)])
    same = pose_geometry_match(solo, solo)
    mirrored = pose_geometry_match(solo, mirror_pose_frame(solo))

    first = person_pose(offset_x=-0.22, bend=0.08)
    second = person_pose(offset_x=0.28, bend=-0.06)
    couple = frame_from_people([first, second])
    reordered_couple = frame_from_people([second, first])
    coupled = pose_geometry_match(couple, reordered_couple)

    assert same.precise is True
    assert mirrored.precise is True
    assert coupled.precise is True
    assert same.precision_ready is True
    assert mirrored.precision_ready is True
    assert coupled.precision_ready is True
    assert same.precision_score == pytest.approx(1.0, abs=1e-6)
    assert mirrored.precision_score == pytest.approx(1.0, abs=1e-6)
    assert coupled.precision_score == pytest.approx(1.0, abs=1e-6)
    serialized = same.as_dict()
    diagnostics = serialized["diagnostics"]
    assert serialized["precise"] is True
    assert serialized["precision_score"] == pytest.approx(1.0, abs=1e-6)
    assert diagnostics["evidence_threshold"] == 0.15
    assert diagnostics["exact_person_count"] is True
    assert diagnostics["body_common_joints"] == 12
    assert diagnostics["core_anchor_joints"] == 4
    assert diagnostics["distal_limb_joints"] == 8
    assert diagnostics["supported_limb_segments"] == 8
    assert diagnostics["body_coverage"] == pytest.approx(1.0)
    assert diagnostics["matched_person_quality"] == pytest.approx(0.9)
    assert diagnostics["supported_articulations"] == 8
    assert diagnostics["articulation_score"] == pytest.approx(1.0)


def test_full_confidence_materially_different_limbs_fail_strict_pose() -> None:
    reference_pose = person_pose()
    crouched_pose = reference_pose.copy()
    crouched_pose[13] = (0.30, 0.62)
    crouched_pose[15] = (0.53, 0.64)
    crouched_pose[14] = (0.71, 0.64)
    crouched_pose[16] = (0.49, 0.65)

    match = pose_geometry_match(
        frame_from_people([reference_pose]),
        frame_from_people([crouched_pose]),
    )

    assert match.reliable is True
    assert match.score > 0.75
    assert match.group_score == pytest.approx(1.0, abs=1e-6)
    assert match.supported_articulations == 8
    assert match.articulation_score < 0.55
    assert match.precision_score < 0.68
    assert match.precision_ready is True
    assert match.precise is False


def test_upper_body_only_evidence_cannot_certify_a_lower_body_pose() -> None:
    reference_pose = person_pose()
    different_legs = reference_pose.copy()
    different_legs[12:17] = np.asarray(
        [
            (0.95, 0.20),
            (0.95, 0.30),
            (0.95, 0.40),
            (0.95, 0.50),
            (0.95, 0.60),
        ],
        dtype=np.float32,
    )
    upper_body_only = np.zeros((17,), dtype=np.float32)
    upper_body_only[[5, 6, 7, 8, 9, 10, 11]] = 0.9

    match = pose_geometry_match(
        frame_from_people([reference_pose], [upper_body_only]),
        frame_from_people([different_legs], [upper_body_only]),
        allow_mirror=False,
    )

    assert match.reliable is True
    assert match.body_common_joints == 7
    assert match.supported_arm_segments == 4
    assert match.supported_leg_segments == 0
    assert match.supported_arm_articulations == 4
    assert match.supported_leg_articulations == 0
    assert match.precision_ready is False
    assert match.precise is False


def test_geometry_recovers_mirror_with_anatomical_left_right_swap() -> None:
    reference = frame_from_people([person_pose(bend=0.10)])
    mirrored_candidate = mirror_pose_frame(reference)

    direct = pose_geometry_match(reference, mirrored_candidate, allow_mirror=False)
    recovered = pose_geometry_match(reference, mirrored_candidate, allow_mirror=True)

    assert recovered.mirrored is True
    assert recovered.score == pytest.approx(1.0, abs=1e-6)
    assert recovered.score > direct.score + 0.04


def test_mirror_selection_prefers_precise_geometry_over_broad_score() -> None:
    reference = frame_from_people([person_pose()])
    candidate_points = np.asarray(
        [
            (0.52598333, 0.00103116),
            (0.47431767, 0.03500512),
            (0.52972382, 0.05612449),
            (0.40849948, 0.05729276),
            (0.62922585, 0.06604739),
            (0.42855254, 0.27139166),
            (0.60385656, 0.23031610),
            (0.34114105, 0.59475112),
            (0.73778933, 0.49734229),
            (0.31179625, 0.55855912),
            (0.84925157, 0.54867351),
            (0.36852688, 0.63012546),
            (0.57604337, 0.58683676),
            (0.38067541, 0.76787132),
            (0.88369548, 0.69137156),
            (0.31091189, 0.97841054),
            (0.69913018, 0.93092066),
        ],
        dtype=np.float32,
    )
    candidate = frame_from_people([candidate_points])

    direct = pose_geometry_match(reference, candidate, allow_mirror=False)
    mirrored = pose_geometry_match(
        reference,
        mirror_pose_frame(candidate),
        allow_mirror=False,
    )
    selected = pose_geometry_match(reference, candidate, allow_mirror=True)

    assert direct.precise is True
    assert mirrored.score > direct.score
    assert mirrored.precise is False
    assert selected.precise is True
    assert selected.mirrored is False
    assert selected.precision_score == pytest.approx(direct.precision_score)


def test_couple_assignment_is_order_independent_and_count_aware() -> None:
    first = person_pose(offset_x=-0.22, bend=0.08)
    second = person_pose(offset_x=0.28, bend=-0.06)
    reference = frame_from_people([first, second])
    reversed_candidate = frame_from_people([second, first])

    matched = pose_geometry_match(reference, reversed_candidate)

    assert matched.score == pytest.approx(1.0, abs=1e-6)
    assert set(matched.matched_pairs) == {(0, 1), (1, 0)}
    assert scene_kind(matched.reference_count) == "couple"

    extra = frame_from_people([second, first, person_pose(offset_x=0.42)])
    with_extra = pose_geometry_match(reference, extra)
    assert with_extra.score < matched.score
    assert with_extra.count_score < 1
    assert scene_kind(with_extra.candidate_count) == "group"


def test_group_geometry_penalizes_wrong_relative_person_layout() -> None:
    base = person_pose()
    reference = frame_from_people([base - (0.28, 0), base + (0.28, 0)])
    same = frame_from_people([base - (0.28, 0), base + (0.28, 0)])
    compressed = frame_from_people([base - (0.06, 0), base + (0.06, 0)])

    correct = pose_geometry_match(reference, same)
    wrong = pose_geometry_match(reference, compressed)

    assert correct.group_score == pytest.approx(1.0, abs=1e-6)
    assert wrong.group_score < correct.group_score
    assert wrong.score < correct.score
    assert correct.precise is True
    assert wrong.precision_score < 0.68
    assert wrong.precision_ready is True
    assert wrong.precise is False


def test_one_correct_person_cannot_hide_a_wrong_couple_member() -> None:
    base = person_pose()
    centered = (base - base.mean(axis=0)) * 0.45
    first = centered + np.asarray((0.27, 0.50), dtype=np.float32)
    second = centered + np.asarray((0.73, 0.50), dtype=np.float32)
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float32)
    rotated_second = (
        (second - second.mean(axis=0)) @ rotation.T
        + second.mean(axis=0)
    )

    match = pose_geometry_match(
        frame_from_people([first, second]),
        frame_from_people([first, rotated_second]),
        allow_mirror=False,
    )

    assert match.precision_ready is True
    assert match.shape_score > 0.50
    assert match.articulation_score == pytest.approx(1.0, abs=1e-6)
    assert match.group_score == pytest.approx(1.0, abs=1e-6)
    assert match.minimum_person_shape_score < 0.01
    assert match.precision_score < 0.10
    assert match.precise is False


def test_skeleton_overlay_is_safe_compact_and_self_contained() -> None:
    frame = frame_from_people([person_pose()])
    svg = skeleton_svg(frame, width=320, height=480)
    uri = skeleton_data_uri(frame, width=320, height=480)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert 'viewBox="0 0 320 480"' in svg
    assert "<line " in svg
    assert "<circle " in svg
    assert "<script" not in svg
    assert uri.startswith("data:image/svg+xml;base64,")
