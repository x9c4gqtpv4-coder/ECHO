from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from batch_color.color import srgb_to_oklab
from batch_color.image_io import make_proxy
from batch_color.masking import get_background_mask
from batch_color.parsing import MulticlassEvidence, mediapipe_multiclass
from batch_color.pose import PoseEvidence, PosePoint, vision_pose_evidence


@dataclass(frozen=True)
class SemanticMasks:
    background: Image.Image
    background_core: Image.Image
    background_transition: Image.Image
    garment: Image.Image
    garment_core: Image.Image
    garment_transition: Image.Image
    skin: Image.Image
    skin_core: Image.Image
    skin_transition: Image.Image
    hair: Image.Image
    hair_core: Image.Image
    accessory_protect: Image.Image
    unknown_person: Image.Image
    conflicts: Image.Image
    backend: str
    diagnostics: dict[str, object]
    probabilities: dict[str, np.ndarray]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0.15)
    if len(xs) < 64:
        raise ValueError("Person mask is too small for semantic region analysis")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    proxy = Image.fromarray(
        np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    )
    return proxy.resize(size, Image.Resampling.BILINEAR)


def _continuous_edit_weight(
    score: np.ndarray,
    safe: np.ndarray,
    *,
    lower: float,
    core: float,
    transition_floor: float,
) -> np.ndarray:
    """Map semantic confidence to edit strength without a core/transition cliff.

    The previous core + scaled-transition representation jumped from roughly
    ``transition_floor * core`` to ``core`` at the core threshold.  On faces,
    that made small confidence differences visible as colour blocks.  This
    ramp reaches full confidence continuously at ``core`` while retaining the
    conservative low-confidence floor and the fail-closed lower cutoff.
    """
    if not 0.0 <= lower < core <= 1.0:
        raise ValueError("Semantic edit thresholds must satisfy 0 <= lower < core <= 1")
    if not 0.0 <= transition_floor <= 1.0:
        raise ValueError("transition_floor must be in 0..1")
    confidence = np.asarray(score, dtype=np.float32)
    protection = np.asarray(safe, dtype=np.float32)
    ramp = np.clip((confidence - lower) / (core - lower), 0.0, 1.0)
    gain = transition_floor + (1.0 - transition_floor) * ramp
    return np.where(
        confidence >= lower,
        confidence * protection * gain,
        0.0,
    ).astype(np.float32)


def _clean_authorized_edit_region(
    candidate: np.ndarray,
    seed: np.ndarray,
    *,
    threshold: float,
    closing_size: int = 5,
    feather_radius: float = 2.5,
) -> np.ndarray:
    """Turn semantic confidence into a uniform interior edit authorization.

    Model confidence decides whether a pixel belongs to a region; it must not
    become a visible per-pixel colour opacity.  Only the inside boundary is
    feathered.  Small confidence pinholes are closed, while large protected
    features such as glasses and lips remain excluded.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("authorization threshold must be in 0..1")
    if closing_size < 1 or closing_size % 2 == 0:
        raise ValueError("closing_size must be a positive odd integer")
    confidence = np.asarray(candidate, dtype=np.float32)
    connected = _seed_connected_support(confidence, np.asarray(seed, dtype=np.float32))
    binary = (confidence >= threshold) & (connected > 0)
    if int(np.count_nonzero(binary)) < 16:
        return np.zeros_like(confidence, dtype=np.float32)
    image = Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L")
    if closing_size > 1:
        image = image.filter(ImageFilter.MaxFilter(closing_size)).filter(
            ImageFilter.MinFilter(closing_size)
        )
    hard = np.asarray(image, dtype=np.float32) / 255.0
    soft = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=feather_radius)), dtype=np.float32
    ) / 255.0
    return np.minimum(hard, soft).astype(np.float32)


def _resize_probability(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (values.shape[1], values.shape[0]) == size:
        return np.asarray(values, dtype=np.float32)
    resized = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F").resize(
        size, Image.Resampling.BILINEAR
    )
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def _skin_candidate(rgb: np.ndarray) -> np.ndarray:
    rgb255 = rgb * 255.0
    r, g, b = rgb255[..., 0], rgb255[..., 1], rgb255[..., 2]
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return (
        (r > 35)
        & (g > 20)
        & (b > 12)
        & (cb > 74)
        & (cb < 132)
        & (cr > 133)
        & (cr < 184)
        & ((r - b) > 8)
        & ((r - g) > 2)
        & (r > 0.90 * g)
        & (r > 0.82 * b)
    )


def _robust_skin_probability(lab: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, int]:
    core = seed >= 0.52
    count = int(np.count_nonzero(core))
    if count < 48:
        # No reliable image-local skin family means there is no colour
        # evidence for authorizing a skin edit.  Returning ones here used to
        # remove the colour prior entirely and could turn coarse parser or
        # geometry support into an edit authorization.
        return np.zeros(lab.shape[:2], dtype=np.float32), count
    pixels = lab[core]
    center = np.median(pixels, axis=0)
    mad = 1.4826 * np.median(np.abs(pixels - center), axis=0)
    scale = np.maximum(mad, np.array([0.045, 0.010, 0.010], dtype=np.float32))
    delta = (lab - center) / scale
    distance = 0.18 * np.square(delta[..., 0]) + np.square(delta[..., 1]) + np.square(delta[..., 2])
    return np.exp(-0.5 * np.clip(distance, 0.0, 36.0)).astype(np.float32), count


def _point_xy(
    point: PosePoint,
    original: tuple[int, int],
    proxy: tuple[int, int],
) -> tuple[float, float]:
    return point.x * proxy[0] / original[0], point.y * proxy[1] / original[1]


def _draw_capsule(
    canvas: Image.Image,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    value: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    xy = [tuple(round(item) for item in start), tuple(round(item) for item in end)]
    draw.line(xy, fill=value, width=width)
    radius = width // 2
    for x, y in xy:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _face_geometry(
    pose: PoseEvidence | None,
    proxy_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object] | None]:
    height, width = proxy_size[1], proxy_size[0]
    image = Image.new("L", proxy_size, 0)
    if pose is None or not pose.faces:
        return np.zeros((height, width), dtype=np.float32), None
    face = max(
        pose.faces,
        key=lambda item: float(item.get("bbox", {}).get("width", 0))
        * float(item.get("bbox", {}).get("height", 0)),
    )
    box = face.get("bbox", {})
    sx, sy = width / pose.width, height / pose.height
    x = float(box.get("x", 0)) * sx
    y = float(box.get("y", 0)) * sy
    w = float(box.get("width", 0)) * sx
    h = float(box.get("height", 0)) * sy
    draw = ImageDraw.Draw(image)
    draw.ellipse((x + 0.17 * w, y + 0.08 * h, x + 0.83 * w, y + 0.91 * h), fill=255)
    landmarks = face.get("landmarks", {}) if isinstance(face.get("landmarks", {}), dict) else {}
    for name in ("leftEye", "rightEye", "leftEyebrow", "rightEyebrow", "outerLips", "innerLips"):
        points = landmarks.get(name, [])
        polygon = [
            (float(point["x"]) * sx, float(point["y"]) * sy)
            for point in points
            if isinstance(point, dict) and "x" in point and "y" in point
        ]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=0)
    # A hard ellipse becomes a visible colour boundary when used as an edit
    # confidence multiplier.  Blur only the geometry evidence; semantic face
    # parsing still protects hair, glasses, eyes and lips.
    radius = max(2.0, 0.025 * min(max(w, 1.0), max(h, 1.0)))
    softened = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(softened, dtype=np.float32) / 255.0, face


def _face_feature_protection(
    pose: PoseEvidence | None,
    face: dict[str, object] | None,
    proxy_size: tuple[int, int],
) -> np.ndarray:
    """Protect eyes, eyebrows and lips from a uniform face colour edit."""
    height, width = proxy_size[1], proxy_size[0]
    image = Image.new("L", proxy_size, 0)
    if pose is None or face is None:
        return np.zeros((height, width), dtype=np.float32)
    landmarks = face.get("landmarks", {}) if isinstance(face.get("landmarks", {}), dict) else {}
    sx, sy = width / pose.width, height / pose.height
    draw = ImageDraw.Draw(image)
    for name in (
        "leftEye",
        "rightEye",
        "leftEyebrow",
        "rightEyebrow",
        "outerLips",
        "innerLips",
    ):
        points = landmarks.get(name, [])
        polygon = [
            (float(point["x"]) * sx, float(point["y"]) * sy)
            for point in points
            if isinstance(point, dict) and "x" in point and "y" in point
        ]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)
    protected = image.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )
    return np.asarray(protected, dtype=np.float32) / 255.0


def _pose_geometry(
    pose: PoseEvidence | None,
    proxy_size: tuple[int, int],
    person_box: tuple[int, int, int, int],
    face: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    height, width = proxy_size[1], proxy_size[0]
    arm = Image.new("L", proxy_size, 0)
    hand = Image.new("L", proxy_size, 0)
    neck = Image.new("L", proxy_size, 0)
    torso = Image.new("L", proxy_size, 0)
    hand_regions: list[np.ndarray] = []
    if pose is None:
        empty = np.zeros((height, width), dtype=np.float32)
        return {
            "arms": empty,
            "hands": empty.copy(),
            "neck": empty.copy(),
            "torso": empty.copy(),
            "hand_regions": hand_regions,
        }, {"body_count": 0, "hand_count": 0, "high_confidence_hand_regions": 0}

    original = (pose.width, pose.height)
    person_width = max(person_box[2] - person_box[0], 1)
    body = max(pose.bodies, key=len) if pose.bodies else {}
    torso_points = [
        body.get("leftShoulder"),
        body.get("rightShoulder"),
        body.get("rightHip"),
        body.get("leftHip"),
    ]
    if all(point is not None and point.confidence >= 0.18 for point in torso_points):
        polygon = [_point_xy(point, original, proxy_size) for point in torso_points]
        ImageDraw.Draw(torso).polygon(polygon, fill=255)
    for side in ("left", "right"):
        shoulder, elbow, wrist = (
            body.get(f"{side}Shoulder"), body.get(f"{side}Elbow"), body.get(f"{side}Wrist")
        )
        for first, second in ((shoulder, elbow), (elbow, wrist)):
            if first and second and min(first.confidence, second.confidence) >= 0.22:
                _draw_capsule(
                    arm,
                    _point_xy(first, original, proxy_size),
                    _point_xy(second, original, proxy_size),
                    max(7, round(person_width * 0.075)),
                    round(255 * min(first.confidence, second.confidence)),
                )

    if face is not None and body.get("neck"):
        box = face.get("bbox", {})
        face_bottom = (
            (float(box.get("x", 0)) + 0.5 * float(box.get("width", 0))) * width / pose.width,
            (float(box.get("y", 0)) + float(box.get("height", 0))) * height / pose.height,
        )
        neck_point = _point_xy(body["neck"], original, proxy_size)
        _draw_capsule(
            neck,
            face_bottom,
            neck_point,
            max(8, round(person_width * 0.13)),
            round(255 * body["neck"].confidence),
        )

    high_confidence_hands = 0
    dilation = max(3, 2 * round(person_width * 0.015) + 1)
    if dilation % 2 == 0:
        dilation += 1
    for points in pose.hands:
        confident = [
            _point_xy(point, original, proxy_size)
            for point in points.values()
            if point.confidence >= 0.22
        ]
        if len(confident) < 5:
            continue
        hull = _convex_hull(confident)
        layer = Image.new("L", proxy_size, 0)
        if len(hull) >= 3:
            ImageDraw.Draw(layer).polygon(hull, fill=255)
        elif len(hull) == 2:
            _draw_capsule(layer, hull[0], hull[1], max(5, round(person_width * 0.04)), 255)
        layer = layer.filter(ImageFilter.MaxFilter(dilation))
        hand = Image.fromarray(
            np.maximum(np.asarray(hand, dtype=np.uint8), np.asarray(layer, dtype=np.uint8)), mode="L"
        )
        hand_regions.append(np.asarray(layer, dtype=np.float32) / 255.0)
        high_confidence_hands += 1

    return {
        "arms": np.asarray(arm, dtype=np.float32) / 255.0,
        "hands": np.asarray(hand, dtype=np.float32) / 255.0,
        "neck": np.asarray(neck, dtype=np.float32) / 255.0,
        "torso": np.asarray(torso, dtype=np.float32) / 255.0,
        "hand_regions": hand_regions,
    }, {
        "body_count": len(pose.bodies),
        "hand_count": len(pose.hands),
        "high_confidence_hand_regions": high_confidence_hands,
    }


def _topology_prior(rel_y: np.ndarray, garment_kind: str) -> np.ndarray:
    ranges = {
        "top": (0.12, 0.76),
        "dress": (0.12, 0.94),
        "bottom": (0.39, 0.98),
        "set": (0.12, 0.98),
    }
    low, high = ranges[garment_kind]
    feather = 0.035
    lower = np.clip((rel_y - low) / feather, 0.0, 1.0)
    upper = np.clip((high - rel_y) / feather, 0.0, 1.0)
    return np.minimum(lower, upper).astype(np.float32)


def _fallback_garment(
    garment_hint: str,
    luminance: np.ndarray,
    chroma: np.ndarray,
) -> np.ndarray:
    if garment_hint == "light":
        return ((luminance > 0.50) & (chroma < 0.20)).astype(np.float32)
    if garment_hint == "dark":
        return (luminance < 0.50).astype(np.float32)
    if garment_hint == "midtone":
        return ((luminance >= 0.30) & (luminance <= 0.78)).astype(np.float32)
    return np.ones_like(luminance, dtype=np.float32)


def _garment_color_prior(
    garment_hint: str,
    luminance: np.ndarray,
    chroma: np.ndarray,
) -> np.ndarray:
    """A soft SKU constraint, never the primary semantic classifier."""
    if garment_hint == "light":
        lightness = np.clip((luminance - 0.38) / 0.36, 0.0, 1.0)
        neutrality = np.clip((0.28 - chroma) / 0.22, 0.0, 1.0)
        return (0.10 + 0.90 * lightness * neutrality).astype(np.float32)
    if garment_hint == "dark":
        darkness = np.clip((0.68 - luminance) / 0.42, 0.0, 1.0)
        return (0.12 + 0.88 * darkness).astype(np.float32)
    if garment_hint == "midtone":
        distance = np.abs(luminance - 0.54)
        return (0.18 + 0.82 * np.clip(1.0 - distance / 0.32, 0.0, 1.0)).astype(np.float32)
    return np.ones_like(luminance, dtype=np.float32)


def _garment_seed_prior(
    lab: np.ndarray,
    seed_weight: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fit one image-local garment colour family from a pose-constrained torso seed."""
    seed = seed_weight >= 0.22
    count = int(np.count_nonzero(seed))
    if count < 96:
        # A missing torso/bottom seed is unavailable evidence, not permission
        # to accept every colour family.
        return np.zeros(lab.shape[:2], dtype=np.float32), count
    pixels = lab[seed]
    center = np.median(pixels, axis=0)
    mad = 1.4826 * np.median(np.abs(pixels - center), axis=0)
    scale = np.maximum(mad, np.array([0.070, 0.014, 0.014], dtype=np.float32))
    delta = (lab - center) / scale
    distance = 0.24 * np.square(delta[..., 0]) + np.square(delta[..., 1]) + np.square(delta[..., 2])
    return np.exp(-0.5 * np.clip(distance, 0.0, 36.0)).astype(np.float32), count


def _seed_connected_support(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep only semantic components touching the pose-constrained garment seed."""
    binary = np.asarray(candidate >= 0.12, dtype=np.uint8)
    touched = np.asarray(seed >= 0.16, dtype=bool) & (binary > 0)
    if int(np.count_nonzero(touched)) < 16:
        # No reliable garment seed means there is no evidence for selecting a
        # component.  Fail closed; returning all pixels would silently turn a
        # local garment mask into a full-frame edit authorization.
        return np.zeros_like(candidate, dtype=np.float32)
    try:
        import cv2

        _, labels = cv2.connectedComponents(binary, connectivity=8)
        keep = np.unique(labels[touched])
        keep = keep[keep != 0]
        return np.isin(labels, keep).astype(np.float32)
    except ImportError:
        height, width = binary.shape
        seen = np.zeros_like(binary, dtype=bool)
        queue: deque[tuple[int, int]] = deque(
            (int(y), int(x)) for y, x in np.argwhere(touched)
        )
        while queue:
            y, x = queue.popleft()
            if seen[y, x] or not binary[y, x]:
                continue
            seen[y, x] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx]:
                        queue.append((ny, nx))
        return seen.astype(np.float32)


def build_semantic_masks(
    input_path: str | Path,
    image: Image.Image,
    *,
    garment_kind: str,
    garment_hint: str,
    mask_backend: str = "vision",
    parser_backend: str = "auto",
    pose_backend: str = "auto",
    proxy_edge: int = 1024,
    enforce_quality: bool = True,
) -> SemanticMasks:
    if garment_kind not in {"none", "top", "dress", "bottom", "set"}:
        raise ValueError("garment_kind must be none, top, dress, bottom, or set")
    if garment_hint not in {"none", "light", "dark", "midtone", "any"}:
        raise ValueError("garment_hint must be none, light, dark, midtone, or any")
    if parser_backend not in {"auto", "mediapipe", "none"}:
        raise ValueError("parser_backend must be auto, mediapipe, or none")
    if pose_backend not in {"auto", "vision", "none"}:
        raise ValueError("pose_backend must be auto, vision, or none")

    background_result = get_background_mask(
        input_path, image, backend=mask_backend, quality="accurate"
    )
    background_full = background_result.background_mask.convert("L")
    proxy = make_proxy(image, max_edge=proxy_edge)
    proxy_size = proxy.size
    background = background_full.resize(proxy_size, Image.Resampling.BILINEAR)
    bg = np.asarray(background, dtype=np.float32) / 255.0
    person = np.clip(1.0 - bg, 0.0, 1.0)
    background_core = np.where(bg >= 0.86, bg, 0.0).astype(np.float32)
    background_transition = np.where(
        (bg >= 0.12) & (bg < 0.86), bg, 0.0
    ).astype(np.float32)
    background_edit = np.clip(background_core + 0.24 * background_transition, 0.0, 1.0)
    x0, y0, x1, y1 = _bbox(person)
    box_h = max(y1 - y0, 1)
    yy, _ = np.mgrid[: person.shape[0], : person.shape[1]]
    rel_y = (yy - y0) / box_h

    rgb = np.asarray(proxy.convert("RGB"), dtype=np.float32) / 255.0
    lab = srgb_to_oklab(rgb)
    luminance = lab[..., 0]
    chroma = np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2]))
    fixed_skin = _skin_candidate(rgb).astype(np.float32)

    parser: MulticlassEvidence | None = None
    parser_warning = None
    if parser_backend != "none":
        try:
            parser = mediapipe_multiclass(proxy)
        except (FileNotFoundError, RuntimeError, OSError) as error:
            if parser_backend == "mediapipe":
                raise
            parser_warning = str(error)

    pose: PoseEvidence | None = None
    pose_warning = None
    if pose_backend != "none":
        try:
            pose = vision_pose_evidence(input_path, canonical_image=image)
        except (FileNotFoundError, RuntimeError, OSError) as error:
            if pose_backend == "vision":
                raise
            pose_warning = str(error)

    face_geo, face_record = _face_geometry(pose, proxy_size)
    face_feature_protect = _face_feature_protection(pose, face_record, proxy_size)
    pose_maps, pose_diagnostics = _pose_geometry(pose, proxy_size, (x0, y0, x1, y1), face_record)
    limb_geo = np.maximum.reduce([pose_maps["arms"], pose_maps["hands"], pose_maps["neck"]])
    torso_geo = pose_maps["torso"]
    if not np.any(torso_geo > 0):
        xx = np.broadcast_to(np.arange(person.shape[1]), person.shape)
        center_x = 0.5 * (x0 + x1)
        half_width = 0.24 * max(x1 - x0, 1)
        torso_geo = (
            (np.abs(xx - center_x) <= half_width)
            & (rel_y >= 0.16)
            & (rel_y <= 0.55)
        ).astype(np.float32)

    if parser is not None:
        face_parser = _resize_probability(parser.face_skin, proxy_size)
        body_parser = _resize_probability(parser.body_skin, proxy_size)
        hair_probability = _resize_probability(parser.hair, proxy_size)
        clothes_probability = _resize_probability(parser.clothes, proxy_size)
        accessory_probability = _resize_probability(parser.accessories, proxy_size)
        parser_name = parser.backend
    else:
        face_parser = fixed_skin * np.maximum(face_geo, 0.25)
        body_parser = fixed_skin * limb_geo
        head_zone = (rel_y > -0.02) & (rel_y < 0.34)
        hair_probability = (
            head_zone & (luminance < 0.68) & (chroma < 0.24) & (fixed_skin < 0.5)
        ).astype(np.float32)
        clothes_probability = _fallback_garment(garment_hint, luminance, chroma)
        accessory_probability = np.zeros_like(person, dtype=np.float32)
        parser_name = "conservative-color-geometry-fallback"

    face_seed = person * np.maximum(face_geo, 0.15 * face_parser) * face_parser
    skin_likelihood, face_seed_pixels = _robust_skin_probability(lab, face_seed)
    face_score = person * face_parser * (0.55 + 0.45 * face_geo) * np.sqrt(skin_likelihood)
    body_score = person * body_parser * (0.42 + 0.58 * limb_geo) * np.sqrt(skin_likelihood)
    skin_score_raw = np.clip(np.maximum(face_score, body_score), 0.0, 1.0)

    # Confidence is used to authorize the face region, not as colour opacity.
    # Otherwise a 256px parser's confidence variation becomes a visible block
    # after full-resolution colour correction.  Dark non-skin structures stay
    # protected even if the coarse face parser leaks across them.
    face_candidate = np.clip(
        person * face_parser * (0.68 + 0.32 * np.sqrt(skin_likelihood)),
        0.0,
        1.0,
    )
    dark_non_skin = (luminance < 0.30) & (fixed_skin < 0.5)
    face_candidate = np.where(dark_non_skin, 0.0, face_candidate).astype(np.float32)
    if face_seed_pixels < 48:
        face_candidate = np.zeros_like(face_candidate, dtype=np.float32)
        face_edit_authorization = np.zeros_like(face_candidate, dtype=np.float32)
    else:
        face_edit_authorization = _clean_authorized_edit_region(
            face_candidate,
            face_seed,
            threshold=0.16,
            closing_size=5,
            feather_radius=2.5,
        )
        face_edit_authorization *= 1.0 - face_feature_protect
    body_candidate = np.clip(
        person * body_parser * (0.68 + 0.32 * np.sqrt(skin_likelihood)),
        0.0,
        1.0,
    )
    body_candidate = np.where(dark_non_skin, 0.0, body_candidate).astype(np.float32)
    if face_seed_pixels < 48:
        body_candidate = np.zeros_like(body_candidate, dtype=np.float32)
        body_edit_authorization = np.zeros_like(body_candidate, dtype=np.float32)
    else:
        body_edit_authorization = _clean_authorized_edit_region(
            body_candidate,
            person * body_parser * limb_geo,
            threshold=0.16,
            closing_size=3,
            feather_radius=2.0,
        )

    hair_score_raw = np.clip(person * hair_probability, 0.0, 1.0)
    accessory_score = np.clip(person * accessory_probability, 0.0, 1.0)
    if garment_kind == "none":
        garment_score_raw = np.zeros_like(person, dtype=np.float32)
        garment_edit_authorization = np.zeros_like(person, dtype=np.float32)
        garment_seed_pixels = 0
    else:
        topology = _topology_prior(rel_y, garment_kind)
        hint_prior = _garment_color_prior(garment_hint, luminance, chroma)
        if garment_kind == "bottom":
            seed_zone = (rel_y >= 0.42) & (rel_y <= 0.72)
            seed_geometry = person * seed_zone.astype(np.float32)
        else:
            seed_geometry = person * torso_geo
        garment_seed = (
            seed_geometry
            * clothes_probability
            * hint_prior
            * (1.0 - 0.90 * skin_score_raw)
            * (1.0 - 0.80 * hair_score_raw)
            * (1.0 - 0.90 * accessory_score)
        )
        self_color_prior, garment_seed_pixels = _garment_seed_prior(lab, garment_seed)
        if garment_seed_pixels < 96:
            garment_score_raw = np.zeros_like(person, dtype=np.float32)
            garment_edit_authorization = np.zeros_like(person, dtype=np.float32)
        else:
            color_prior = hint_prior * (0.08 + 0.92 * self_color_prior)
            garment_score_raw = np.clip(
                person
                * topology
                * clothes_probability
                * color_prior
                * (1.0 - 0.80 * skin_score_raw)
                * (1.0 - 0.75 * hair_score_raw)
                * (1.0 - 0.85 * accessory_score),
                0.0,
                1.0,
            )
            connected = _seed_connected_support(garment_score_raw, garment_seed)
            garment_score_raw *= connected
            # Parser and colour-model confidence authorize the garment; they
            # must not become a per-thread opacity map.  On pale knitwear the
            # previous confidence-weighted mask followed every rib and shadow,
            # so the white garment was only corrected in alternating stripes.
            # Close narrow textile gaps into one continuous garment interior,
            # while the connected seed, semantic conflicts and feathered edge
            # still keep skin, background and unrelated trousers protected.
            garment_edit_authorization = _clean_authorized_edit_region(
                garment_score_raw,
                garment_seed,
                threshold=0.12,
                closing_size=7,
                feather_radius=2.5,
            )

    conflict_score = np.maximum.reduce(
        [
            np.minimum(skin_score_raw, garment_score_raw),
            np.minimum(hair_score_raw, garment_score_raw),
            np.minimum(accessory_score, garment_score_raw),
        ]
    )
    conflict_core = np.where(conflict_score >= 0.25, conflict_score, 0.0).astype(np.float32)
    safe = 1.0 - np.clip(conflict_core * 1.6, 0.0, 1.0)

    skin_core = np.where(skin_score_raw >= 0.55, skin_score_raw * safe, 0.0).astype(np.float32)
    skin_transition = np.where(
        (skin_score_raw >= 0.16) & (skin_score_raw < 0.55), skin_score_raw * safe, 0.0
    ).astype(np.float32)
    garment_core = np.where(
        garment_score_raw >= 0.56, garment_score_raw * safe, 0.0
    ).astype(np.float32)
    garment_transition = np.where(
        (garment_score_raw >= 0.14) & (garment_score_raw < 0.56),
        garment_score_raw * safe,
        0.0,
    ).astype(np.float32)
    hair_core = np.where(hair_score_raw >= 0.55, hair_score_raw * safe, 0.0).astype(np.float32)

    skin_edit = np.clip(
        np.maximum.reduce(
            [
                _continuous_edit_weight(
                    skin_score_raw,
                    safe,
                    lower=0.16,
                    core=0.55,
                    transition_floor=0.38,
                ),
                face_edit_authorization * safe,
                body_edit_authorization * safe,
            ]
        )
        * (1.0 - face_feature_protect),
        0.0,
        1.0,
    )
    garment_edit = np.clip(
        np.maximum(
            _continuous_edit_weight(
                garment_score_raw,
                safe,
                lower=0.14,
                core=0.56,
                transition_floor=0.30,
            ),
            garment_edit_authorization * safe,
        ),
        0.0,
        1.0,
    )
    hair_edit = np.clip(hair_core, 0.0, 1.0)
    claimed = np.maximum.reduce(
        [skin_core, garment_core, hair_core, np.where(accessory_score >= 0.48, accessory_score, 0.0)]
    )
    unknown = np.clip(person * (1.0 - claimed), 0.0, 1.0)

    garment_pixels = int(np.count_nonzero(garment_core >= 0.56))
    skin_pixels = int(np.count_nonzero(skin_core >= 0.55))
    hair_pixels = int(np.count_nonzero(hair_core >= 0.55))
    person_pixels = max(int(np.count_nonzero(person >= 0.30)), 1)
    conflict_ratio = float(np.count_nonzero(conflict_core > 0) / person_pixels)
    skin_ratio = float(np.count_nonzero(skin_core > 0) / person_pixels)
    garment_rectangularity = 0.0
    if garment_pixels:
        gx0, gy0, gx1, gy1 = _bbox(garment_core)
        garment_rectangularity = float(garment_pixels / max((gx1 - gx0) * (gy1 - gy0), 1))

    hand_coverages = []
    for region in pose_maps.get("hand_regions", []):
        visible = region >= 0.30
        if np.any(visible):
            hand_coverages.append(float(np.mean(skin_score_raw[visible] >= 0.30)))

    warnings = []
    fatal_flags = []
    if parser is None:
        warnings.append("multiclass_parser_unavailable_review_only")
    if pose is None:
        warnings.append("pose_evidence_unavailable")
    if face_seed_pixels < 48:
        warnings.append("skin_colour_seed_insufficient_edit_disabled")
    if garment_kind != "none" and garment_seed_pixels < 96:
        warnings.append("garment_colour_seed_insufficient_edit_disabled")
    if face_record is not None and face_seed_pixels < 48:
        fatal_flags.append("detected_face_without_reliable_skin_seed")
    if garment_kind != "none" and garment_pixels < 256:
        fatal_flags.append("garment_core_too_small")
    if garment_kind != "none" and garment_rectangularity > 0.94:
        fatal_flags.append("garment_mask_rectangular_degeneration")
    if face_record is not None and face_seed_pixels >= 48 and skin_pixels < 32:
        fatal_flags.append("detected_face_without_skin_core")
    if conflict_ratio > 0.08:
        fatal_flags.append("semantic_conflict_excessive")
    if hand_coverages and min(hand_coverages) < 0.08:
        warnings.append("high_confidence_hand_without_skin_support")
    if conflict_ratio > 0.02:
        warnings.append("semantic_conflict_requires_review")
    if fatal_flags and enforce_quality:
        raise ValueError(
            f"Semantic mask quality gate failed for {Path(input_path).name}: "
            + ", ".join(fatal_flags)
        )

    size = image.size
    probabilities = {
        "background_score": bg.astype(np.float32),
        "person": person.astype(np.float32),
        "face_geometry": face_geo.astype(np.float32),
        "arm_geometry": pose_maps["arms"].astype(np.float32),
        "hand_geometry": pose_maps["hands"].astype(np.float32),
        "neck_geometry": pose_maps["neck"].astype(np.float32),
        "skin_likelihood": skin_likelihood.astype(np.float32),
        "skin_score": skin_score_raw.astype(np.float32),
        "face_edit_authorization": face_edit_authorization.astype(np.float32),
        "body_edit_authorization": body_edit_authorization.astype(np.float32),
        "face_feature_protection": face_feature_protect.astype(np.float32),
        "garment_score": garment_score_raw.astype(np.float32),
        "garment_edit_authorization": garment_edit_authorization.astype(np.float32),
        "hair_score": hair_score_raw.astype(np.float32),
        "accessory_score": accessory_score.astype(np.float32),
        "conflict_score": conflict_score.astype(np.float32),
    }
    return SemanticMasks(
        background=_mask(background_edit, size),
        background_core=_mask(background_core, size),
        background_transition=_mask(background_transition, size),
        garment=_mask(garment_edit, size),
        garment_core=_mask(garment_core, size),
        garment_transition=_mask(garment_transition, size),
        skin=_mask(skin_edit, size),
        skin_core=_mask(skin_core, size),
        skin_transition=_mask(skin_transition, size),
        hair=_mask(hair_edit, size),
        hair_core=_mask(hair_core, size),
        accessory_protect=_mask(accessory_score, size),
        unknown_person=_mask(unknown, size),
        conflicts=_mask(conflict_core, size),
        backend=background_result.backend,
        diagnostics={
            "proxy_size": list(proxy_size),
            "person_bbox_proxy": [x0, y0, x1, y1],
            "garment_core_pixels_proxy": garment_pixels,
            "skin_core_pixels_proxy": skin_pixels,
            "hair_core_pixels_proxy": hair_pixels,
            "face_seed_pixels_proxy": face_seed_pixels,
            "face_edit_authorized_pixels_proxy": int(
                np.count_nonzero(face_edit_authorization >= 0.50)
            ),
            "body_edit_authorized_pixels_proxy": int(
                np.count_nonzero(body_edit_authorization >= 0.50)
            ),
            "garment_seed_pixels_proxy": garment_seed_pixels,
            "skin_person_ratio": round(skin_ratio, 6),
            "semantic_conflict_ratio": round(conflict_ratio, 6),
            "garment_rectangularity": round(garment_rectangularity, 6),
            "hand_skin_coverages": [round(value, 6) for value in hand_coverages],
            "fatal_flags": fatal_flags,
            "warnings": warnings,
            "garment_kind": garment_kind,
            "garment_hint": garment_hint,
            "semantic_backend": parser_name,
            "pose_backend": pose.backend if pose else "unavailable",
            "parser_warning": parser_warning,
            "pose_warning": pose_warning,
            "human_review_required": True,
            **pose_diagnostics,
        },
        probabilities=probabilities,
    )
