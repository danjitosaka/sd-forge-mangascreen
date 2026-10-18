"""
MangaScreen Core Engine for SD WebUI Forge Neo / A1111
Ported and extended from https://github.com/ashukae/ComfyUI-MangaScreen

Implements:
- Pre-scaling to target working megapixels (Lanczos)
- Manga Levels (grayscale + black/white/gamma level adjustment)
- Automatic Pen Line extraction (Manga Line Extraction erika.pth / Anime Lineart netG.pth / Custom)
- Manga Finish (line separation, normalized convolution descreening, 4-level tone plan,
  gradation tone detection, 3x supersampled 45-degree AM halftone screen, white cut inside solid black)
- Manga Panel (aspect crop, panel border, page margin, paper color/fibre, printed ink bleed)
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

ANGLE_DEFAULT = 45.0

FRAMES = {
    "original": None,
    "4:3 wide": 4.0 / 3.0,
    "16:9 wide": 16.0 / 9.0,
    "3:4 tall": 3.0 / 4.0,
    "1:1": 1.0,
}

PAPERS = {
    "white": ((255, 255, 255), 0.0),
    "off-white": ((251, 250, 247), 1.5),
    "cream": ((246, 243, 236), 3.0),
}

INK_RGB = np.float32((26, 25, 28))

PEN_MODES = [
    "Manga Lineart (erika.pth - Recommended)",
    "Anime Lineart (netG.pth)",
    "Internal OpenCV Lines Only (Zero VRAM)",
    "Custom Line Map (External Image)",
]

PEN_SOURCES = [
    "From Manga Levels Output (Recommended)",
    "From Original Image",
]

PRESETS = {
    "Manga-Lite (Color Illustration -> Manga / Universal)": {
        "scale_mp": 2.0,
        "enable_levels": True,
        "levels_black": 0.20,
        "levels_white": 0.70,
        "levels_gamma": 1.30,
        "pen_mode": "Manga Lineart (erika.pth - Recommended)",
        "pen_source": "From Manga Levels Output (Recommended)",
        "pen_resolution": 1280,
        "enable_finish": True,
        "screen_lines": 250,
        "white_point": 0.80,
        "black_point": 0.30,
        "light_tone": 0.18,
        "dark_tone": 0.42,
        "gradation": 1.00,
        "pen_threshold": 0.30,
        "white_lines": 0.40,
        "min_region": 600,
        "screen_angle": 45.0,
        "enable_panel": True,
        "frame": "original",
        "crop_x": 0.50,
        "crop_y": 0.40,
        "border": 5,
        "margin": 60,
        "paper": "white",
        "ink": 0.50,
    },
    "Bold Beta / Dark Clothes (Solid Black Ink with White Folds)": {
        "scale_mp": 2.0,
        "enable_levels": True,
        "levels_black": 0.20,
        "levels_white": 0.72,
        "levels_gamma": 1.25,
        "pen_mode": "Manga Lineart (erika.pth - Recommended)",
        "pen_source": "From Manga Levels Output (Recommended)",
        "pen_resolution": 1280,
        "enable_finish": True,
        "screen_lines": 250,
        "white_point": 0.80,
        "black_point": 0.50,
        "light_tone": 0.18,
        "dark_tone": 0.45,
        "gradation": 1.00,
        "pen_threshold": 0.28,
        "white_lines": 0.55,
        "min_region": 600,
        "screen_angle": 45.0,
        "enable_panel": True,
        "frame": "original",
        "crop_x": 0.50,
        "crop_y": 0.40,
        "border": 5,
        "margin": 60,
        "paper": "white",
        "ink": 0.50,
    },
    "Pre-Inked B&W / Qwen-Manga Edit (Skip Pre-Levels)": {
        "scale_mp": 2.0,
        "enable_levels": False,
        "levels_black": 0.20,
        "levels_white": 0.70,
        "levels_gamma": 1.30,
        "pen_mode": "Manga Lineart (erika.pth - Recommended)",
        "pen_source": "From Original Image",
        "pen_resolution": 1280,
        "enable_finish": True,
        "screen_lines": 250,
        "white_point": 0.80,
        "black_point": 0.35,
        "light_tone": 0.18,
        "dark_tone": 0.42,
        "gradation": 1.00,
        "pen_threshold": 0.30,
        "white_lines": 0.40,
        "min_region": 600,
        "screen_angle": 45.0,
        "enable_panel": True,
        "frame": "original",
        "crop_x": 0.50,
        "crop_y": 0.40,
        "border": 5,
        "margin": 60,
        "paper": "white",
        "ink": 0.50,
    },
    "Frameless Full Page (No Border / No Margin)": {
        "scale_mp": 2.0,
        "enable_levels": True,
        "levels_black": 0.20,
        "levels_white": 0.70,
        "levels_gamma": 1.30,
        "pen_mode": "Manga Lineart (erika.pth - Recommended)",
        "pen_source": "From Manga Levels Output (Recommended)",
        "pen_resolution": 1280,
        "enable_finish": True,
        "screen_lines": 220,
        "white_point": 0.80,
        "black_point": 0.30,
        "light_tone": 0.18,
        "dark_tone": 0.42,
        "gradation": 1.00,
        "pen_threshold": 0.30,
        "white_lines": 0.40,
        "min_region": 600,
        "screen_angle": 45.0,
        "enable_panel": False,
        "frame": "original",
        "crop_x": 0.50,
        "crop_y": 0.40,
        "border": 0,
        "margin": 0,
        "paper": "white",
        "ink": 0.00,
    },
}


def scale_to_megapixels(img: np.ndarray, megapixels: float, step: int = 32) -> np.ndarray:
    """Scale RGB uint8 image to target total megapixels using Lanczos interpolation."""
    if megapixels is None or megapixels <= 0:
        return img
    h, w = img.shape[:2]
    current_pixels = float(h * w)
    target_pixels = float(megapixels) * 1_000_000.0
    if abs(current_pixels - target_pixels) / target_pixels < 0.02:
        return img
    scale = np.sqrt(target_pixels / max(current_pixels, 1.0))
    new_w = max(step, int(round((w * scale) / step) * step))
    new_h = max(step, int(round((h * scale) / step) * step))
    if new_w == w and new_h == h:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def apply_manga_levels(img: np.ndarray, black: float = 0.20, white: float = 0.70, gamma: float = 1.30) -> np.ndarray:
    """Grayscale + levels: brightens until skin and light colours turn to paper white.
    Input and output are RGB uint8 (0..255).
    """
    rgb_f = img[..., :3].astype(np.float32) / 255.0
    g = (
        rgb_f[..., 0] * 0.299
        + rgb_f[..., 1] * 0.587
        + rgb_f[..., 2] * 0.114
    )
    denom = max(float(white) - float(black), 1e-3)
    g = np.clip((g - float(black)) / denom, 0.0, 1.0) ** (1.0 / max(float(gamma), 1e-3))
    out_u8 = (g * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([out_u8, out_u8, out_u8], axis=-1)


def _ensure_legacy_preprocessors_on_path():
    """Ensure Forge's built-in forge_legacy_preprocessors is importable."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    legacy_dir = os.path.join(root_dir, "extensions-builtin", "forge_legacy_preprocessors")
    if os.path.isdir(legacy_dir) and legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)


def extract_pen_lines(
    img: np.ndarray,
    pen_mode: str,
    resolution: int = 1280,
    custom_pen_img: np.ndarray | Image.Image | None = None,
) -> np.ndarray | None:
    """Extract white-on-black line strength map in float32 (0..1) resized to img (H, W)."""
    h, w = img.shape[:2]
    res = int(resolution) if (resolution and int(resolution) > 0) else w

    if not pen_mode or "OpenCV" in pen_mode:
        return None

    if "Custom" in pen_mode:
        if custom_pen_img is None:
            return None
        if isinstance(custom_pen_img, Image.Image):
            custom_np = np.array(custom_pen_img.convert("RGB"), dtype=np.uint8)
        else:
            custom_np = custom_pen_img[..., :3].astype(np.uint8)
        gray = custom_np.mean(axis=-1).astype(np.float32) / 255.0
        # Auto-detect black-on-white lineart vs white-on-black lineart
        if gray.mean() > 0.5:
            gray = 1.0 - gray
        return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    preproc_key = "lineart_anime" if "netG" in pen_mode else "lineart_anime_denoise"

    line_map_u8 = None
    try:
        from modules_forge.shared import supported_preprocessors
        if preproc_key in supported_preprocessors:
            line_map_u8 = supported_preprocessors[preproc_key](img, resolution=res)
    except Exception:
        line_map_u8 = None

    if line_map_u8 is None:
        try:
            _ensure_legacy_preprocessors_on_path()
            from legacy_preprocessors.preprocessor import (
                lineart_anime,
                lineart_anime_denoise,
                unload_lineart_anime,
                unload_lineart_anime_denoise,
            )
            if preproc_key == "lineart_anime":
                line_map_u8, _ = lineart_anime(img, res=res)
                unload_lineart_anime()
            else:
                line_map_u8, _ = lineart_anime_denoise(img, res=res)
                unload_lineart_anime_denoise()
        except Exception as e:
            print(f"[MangaScreen] Auxiliary lineart preprocessor unavailable ({e}); falling back to internal OpenCV lines.")
            return None

    if line_map_u8.ndim == 3:
        m = line_map_u8[..., :3].mean(axis=-1).astype(np.float32) / 255.0
    else:
        m = line_map_u8.astype(np.float32) / 255.0

    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


def print_screen(cover: np.ndarray, pitch: float, angle: float = ANGLE_DEFAULT) -> np.ndarray:
    """Page-aligned AM dots for a coverage map, rendered 3x supersampled so dots are round."""
    h, w = cover.shape
    s = 3
    pitch = max(float(pitch), 1.0)
    y, x = np.mgrid[0 : h * s, 0 : w * s].astype(np.float32)
    a = np.deg2rad(float(angle))
    u = (x * np.cos(a) + y * np.sin(a)) / (pitch * s)
    v = (-x * np.sin(a) + y * np.cos(a)) / (pitch * s)
    spot = 1.0 - (np.cos(2.0 * np.pi * u) + np.cos(2.0 * np.pi * v) + 2.0) / 4.0
    c = cv2.resize(cover, (w * s, h * s), interpolation=cv2.INTER_LINEAR)
    return cv2.resize((spot < c).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5


def ink_layer(L: np.ndarray, k: float) -> np.ndarray:
    """Lines and solid blacks as a crisp mask. k is the image scale relative to 1024 px width."""
    solid = cv2.GaussianBlur(L, (0, 0), max(1.5 * k, 0.1)) < 0.22
    Lb = cv2.GaussianBlur(L, (0, 0), max(0.7 * k, 0.1))
    k_blur = max(int(15 * k) | 1, 3)
    line = (Lb < cv2.blur(Lb, (k_blur, k_blur)) - 0.07) & (Lb < 0.65)
    _, lab, st, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    w, h, area = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    # a stroke is long and thin; a dot, or a clump of generated dots, is compact
    keep = (np.maximum(w, h) >= 10 * k) & (area >= 12 * k * k) & (area < 0.6 * w * h + 40 * k * k)
    keep |= area >= 400 * k * k
    keep[0] = False
    ink = solid | keep[lab]
    _, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = st[:, cv2.CC_STAT_AREA] >= 6 * k * k
    big[0] = False
    return big[lab]


def value_map(L: np.ndarray, ink: np.ndarray, k: float) -> np.ndarray:
    """Descreened value with the ink masked out (normalized convolution), so lines leave no dark
    halo for the screen to pick up, then edge-preserving flattening."""
    m = (~ink).astype(np.float32)
    s = max(2.5 * k, 0.2)
    V = cv2.GaussianBlur(L * m, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)
    d = max(int(9 * k) | 1, 3)
    for _ in range(2):
        V = cv2.bilateralFilter(V, d, 0.10, max(9 * k, 1.0))
    return V


def tone_plan(V: np.ndarray, k: float, white: float, split: float, black: float, min_region: int = 600) -> np.ndarray:
    """0 white, 1 light tone, 2 dark tone, 3 black. Regions under min_region px (at 1024 px width)
    melt into their surroundings, so every tone is a deliberate flat area."""
    k11 = max(int(11 * k) | 1, 3)
    k21 = max(int(21 * k) | 1, 3)
    level = cv2.medianBlur(np.digitize(-V, [-white, -split, -black]).astype(np.uint8), k11)
    for lv in (1, 2, 3, 0):
        _, lab, st, _ = cv2.connectedComponentsWithStats((level == lv).astype(np.uint8), connectivity=4)
        small = st[:, cv2.CC_STAT_AREA] < min_region * k * k
        small[0] = False
        hole = small[lab]
        if hole.any():
            fill = cv2.medianBlur(np.where(hole, 255, level).astype(np.uint8), k21)
            level = np.where(hole & (fill != 255), fill, level).astype(np.uint8)
    return level


def region_spread(level: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standard deviation of V inside each flat region of the plan: a region that hides a gradient spreads wide."""
    out = np.zeros_like(V)
    for lv in range(4):
        n, lab = cv2.connectedComponents((level == lv).astype(np.uint8), connectivity=4)
        cnt = np.maximum(np.bincount(lab.ravel(), minlength=n), 1)
        mean = np.bincount(lab.ravel(), V.ravel(), minlength=n) / cnt
        std = np.sqrt(np.maximum(np.bincount(lab.ravel(), (V * V).ravel(), minlength=n) / cnt - mean * mean, 0))
        out = np.where(level == lv, std[lab], out)
    return out.astype(np.float32)


def white_cut(candidates: np.ndarray, k: float) -> np.ndarray:
    """The strokes worth keeping in white inside beta: long ones only, thinned, so a dark coat shows
    its main folds and seams instead of looking like an inverted line drawing."""
    _, lab, st, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    keep = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= 40 * k
    keep[0] = False
    return cv2.erode(keep[lab].astype(np.uint8), np.ones((2 if k < 1.6 else 3,) * 2, np.uint8)) > 0


def finish(
    img: np.ndarray,
    lines: int = 200,
    white: float = 0.80,
    black: float = 0.30,
    light_tone: float = 0.18,
    dark_tone: float = 0.42,
    gradation: float = 1.0,
    pen: np.ndarray | None = None,
    pen_threshold: float = 0.30,
    white_lines: float = 0.40,
    min_region: int = 600,
    angle: float = ANGLE_DEFAULT,
) -> np.ndarray:
    """Run MangaFinish and return a boolean mask where True = paper (white) and False = ink (black)."""
    white = float(white)
    black = min(float(black), white - 0.02)
    L = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    k = max(L.shape[1] / 1024.0, 0.25)
    ink = ink_layer(L, k)
    strokes = ink & ~(cv2.GaussianBlur(L, (0, 0), max(1.5 * k, 0.1)) < 0.22)  # lines without the solid masses
    if pen is not None:
        strokes |= pen > float(pen_threshold)
    V = value_map(L, ink, k)
    level = tone_plan(V, k, white, (white + black) / 2.0, black, min_region=int(min_region))
    cover = np.take(np.float32((0.0, float(light_tone), float(dark_tone), 1.0)), level)
    if gradation > 0:
        Vs = cv2.GaussianBlur(V, (0, 0), max(3.0 * k, 0.2))
        a = cv2.GaussianBlur(
            np.clip((region_spread(level, Vs) - 0.03) / 0.05, 0.0, 1.0) * float(gradation),
            (0, 0),
            max(4.0 * k, 0.2),
        )
        cover = cover * (1.0 - a) + np.clip((white - Vs) / max(white - black, 1e-3), 0.0, 1.0) * a
    page = print_screen(cover, L.shape[1] / max(int(lines), 10), angle=angle) | ink | strokes
    if white_lines > 0:
        k3 = max(int(3 * k) | 1, 3)
        beta = cv2.erode((level == 3).astype(np.uint8), np.ones((k3, k3), np.uint8)) > 0
        candidates = strokes if pen is None else (pen > (1.0 - float(white_lines)))
        page = (page | beta) & ~(beta & white_cut(candidates, k))
    return ~page


def _noise(h: int, w: int, scales: list[tuple[int, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for s, amp in scales:
        s_clamped = max(1, int(s))
        n = rng.standard_normal((max(2, h // s_clamped), max(2, w // s_clamped))).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def cut_panel(ink: np.ndarray, ratio: float | None, x: float, y: float, border: int, margin: int) -> np.ndarray:
    """Crop the largest window of the given aspect at (x, y), draw the panel border, add the page margin."""
    h, w = ink.shape
    cw, ch = (
        (w, h)
        if ratio is None
        else ((w, round(w / ratio)) if w / h <= ratio else (round(h * ratio), h))
    )
    cw, ch = min(cw, w), min(ch, h)
    x0, y0 = round((w - cw) * float(x)), round((h - ch) * float(y))
    p = ink[y0 : y0 + ch, x0 : x0 + cw].copy()
    if border > 0:
        p[:border] = p[-border:] = True
        p[:, :border] = p[:, -border:] = True
    if margin > 0:
        p = np.pad(p, margin, constant_values=False)
    return p


def print_look(ink: np.ndarray, paper: str, strength: float, k: float) -> np.ndarray:
    """Ink spreads a fraction of a pixel and its density mottles a little; paper colour and fibre."""
    h, w = ink.shape
    a = ink.astype(np.float32)
    strength = float(strength)
    if strength > 0:
        a = np.clip((cv2.GaussianBlur(a, (0, 0), max(0.6 * k * strength, 0.1)) - 0.35) / 0.4, 0.0, 1.0)
        a *= 1.0 - 0.05 * strength * (
            0.5 + 0.5 * np.tanh(_noise(h, w, [(max(1, int(64 * k)), 1.0), (max(1, int(16 * k)), 0.5)], 1))
        )
    rgb, fibre = PAPERS.get(paper, PAPERS["white"])
    base = np.float32(rgb)[None, None] + fibre * _noise(h, w, [(3, 1.0), (1, 0.6)], 2)[..., None]
    return np.clip(base * (1.0 - a[..., None]) + INK_RGB * strength * a[..., None], 0.0, 255.0)


def process_manga_pipeline(
    pil_image: Image.Image,
    *,
    scale_mp: float = 0.0,
    enable_levels: bool = True,
    levels_black: float = 0.20,
    levels_white: float = 0.70,
    levels_gamma: float = 1.30,
    pen_mode: str = "Manga Lineart (erika.pth - Recommended)",
    pen_source: str = "From Manga Levels Output (Recommended)",
    pen_resolution: int = 1280,
    custom_pen_image: Image.Image | None = None,
    enable_finish: bool = True,
    screen_lines: int = 250,
    white_point: float = 0.80,
    black_point: float = 0.30,
    light_tone: float = 0.18,
    dark_tone: float = 0.42,
    gradation: float = 1.00,
    pen_threshold: float = 0.30,
    white_lines: float = 0.40,
    min_region: int = 600,
    screen_angle: float = 45.0,
    enable_panel: bool = True,
    frame: str = "original",
    crop_x: float = 0.50,
    crop_y: float = 0.40,
    border: int = 5,
    margin: int = 60,
    paper: str = "white",
    ink: float = 0.50,
    return_intermediates: bool = False,
) -> tuple[Image.Image, dict[str, Image.Image]]:
    """Execute the full MangaScreen pipeline on a PIL image."""
    img_orig = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    img_scaled = scale_to_megapixels(img_orig, scale_mp)

    intermediates: dict[str, Image.Image] = {}

    if enable_levels:
        img_levels = apply_manga_levels(img_scaled, black=levels_black, white=levels_white, gamma=levels_gamma)
        if return_intermediates:
            intermediates["1. Manga Levels"] = Image.fromarray(img_levels)
    else:
        img_levels = img_scaled

    pen_input = img_levels if "Levels" in str(pen_source) else img_scaled
    pen_map = None
    if enable_finish and "OpenCV" not in str(pen_mode):
        pen_map = extract_pen_lines(
            pen_input,
            pen_mode=pen_mode,
            resolution=pen_resolution,
            custom_pen_img=custom_pen_image,
        )
        if return_intermediates and pen_map is not None:
            pen_vis = (pen_map * 255.0).clip(0, 255).astype(np.uint8)
            intermediates["2. Pen Lines (Stroke Map)"] = Image.fromarray(np.stack([pen_vis] * 3, axis=-1))

    if enable_finish:
        page_white_mask = finish(
            img_levels,
            lines=screen_lines,
            white=white_point,
            black=black_point,
            light_tone=light_tone,
            dark_tone=dark_tone,
            gradation=gradation,
            pen=pen_map,
            pen_threshold=pen_threshold,
            white_lines=white_lines,
            min_region=min_region,
            angle=screen_angle,
        )
        finish_u8 = np.where(page_white_mask[..., None], 255, 0).astype(np.uint8)
        finish_u8 = np.repeat(finish_u8, 3, axis=-1)
        if return_intermediates and enable_panel:
            intermediates["3. Manga Finish (Pre-Panel)"] = Image.fromarray(finish_u8)
    else:
        finish_u8 = img_levels
        page_white_mask = finish_u8[..., :3].mean(axis=-1) >= 128.0

    if enable_panel:
        ink_mask = ~page_white_mask
        k = max(ink_mask.shape[1] / 1024.0, 0.25)
        framed = cut_panel(
            ink_mask,
            FRAMES.get(frame, None),
            crop_x,
            crop_y,
            round(int(border) * k),
            round(int(margin) * k),
        )
        panel_rgb = print_look(framed, paper, ink, k).clip(0, 255).astype(np.uint8)
        final_pil = Image.fromarray(panel_rgb)
    else:
        final_pil = Image.fromarray(finish_u8)

    return final_pil, intermediates
