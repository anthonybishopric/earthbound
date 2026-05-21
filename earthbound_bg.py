#!/usr/bin/env -S .venv/bin/python3
"""
EarthBound-style battle background generator.

Generates animated backgrounds using the same distortion algorithm as the
SNES game: sinusoidal per-scanline offsets applied to a tiling texture,
with optional palette cycling. Outputs to MP4 via ffmpeg.

Usage:
    python earthbound_bg.py [--preset NAME] [--duration SECONDS] [--output FILE]

Presets: fire, ocean, cosmic, acid, cave
"""

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    from imageio_ffmpeg import get_ffmpeg_exe
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

SNES_WIDTH = 256
SNES_HEIGHT = 224
FPS = 30


class DistortionType(IntEnum):
    HORIZONTAL = 1
    HORIZONTAL_INTERLACED = 2
    VERTICAL = 3


@dataclass
class DistortionEffect:
    type: DistortionType
    frequency: int
    amplitude: int
    compression: int
    frequency_accel: int
    amplitude_accel: int
    speed: int
    compression_accel: int


@dataclass
class LayerConfig:
    """Configuration for one background layer."""
    effect: DistortionEffect
    palette: list  # list of (r, g, b) tuples
    palette_cycle_speed: int  # frames between palette shifts (0 = no cycling)
    alpha: float  # blend alpha [0, 1]
    texture_func: str  # name of procedural texture generator
    scroll_x: float = 0.0  # pixels per frame to scroll horizontally (positive = left)
    scroll_y: float = 0.0  # pixels per frame to scroll vertically (positive = up)


# --- Procedural texture generators ---
# Each returns a 256x256 RGB numpy array (uint8)

def gen_horizontal_stripes(palette: list) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    n = len(palette)
    for y in range(256):
        color = palette[y * n // 256]
        img[y, :] = color
    return img


def gen_diagonal_stripes(palette: list) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    n = len(palette)
    for y in range(256):
        for x in range(256):
            idx = ((x + y) * n // 256) % n
            img[y, x] = palette[idx]
    return img


def gen_concentric(palette: list) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    n = len(palette)
    cx, cy = 128, 128
    for y in range(256):
        for x in range(256):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            idx = int(dist * n / 180) % n
            img[y, x] = palette[idx]
    return img


def gen_checkerboard(palette: list) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    n = len(palette)
    for y in range(256):
        for x in range(256):
            block = (x // 16 + y // 16) % n
            img[y, x] = palette[block]
    return img


def gen_plasma(palette: list) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    n = len(palette)
    for y in range(256):
        for x in range(256):
            v = math.sin(x / 16.0)
            v += math.sin(y / 8.0)
            v += math.sin((x + y) / 16.0)
            v += math.sin(math.sqrt(x * x + y * y) / 8.0)
            idx = int((v + 4) * n / 8) % n
            img[y, x] = palette[idx]
    return img


TEXTURE_GENERATORS = {
    "horizontal_stripes": gen_horizontal_stripes,
    "diagonal_stripes": gen_diagonal_stripes,
    "concentric": gen_concentric,
    "checkerboard": gen_checkerboard,
    "plasma": gen_plasma,
}


# --- Distortion engine (port of the SNES algorithm) ---

C1 = 1.0 / 512.0
C2 = 8.0 * math.pi / (1024.0 * 256.0)
C3 = math.pi / 60.0


def compute_frame(
    source: np.ndarray,
    tick: int,
    effect: DistortionEffect,
    alpha: float,
    letterbox: int = 0,
) -> np.ndarray:
    """Apply one frame of distortion to source texture, returning SNES-sized frame."""
    t2 = tick * 2
    amplitude = C1 * (effect.amplitude + effect.amplitude_accel * t2)
    frequency = C2 * (effect.frequency + effect.frequency_accel * t2)
    compression = 1.0 + (effect.compression + effect.compression_accel * t2) / 256.0
    speed = C3 * effect.speed * tick

    dst = np.zeros((SNES_HEIGHT, SNES_WIDTH, 3), dtype=np.float64)
    src_h, src_w = source.shape[:2]

    for y in range(SNES_HEIGHT):
        if y < letterbox or y > SNES_HEIGHT - 1 - letterbox:
            continue

        raw_offset = round(amplitude * math.sin(frequency * y + speed))

        if effect.type == DistortionType.HORIZONTAL:
            offset_x = raw_offset
            src_y = y % src_h
        elif effect.type == DistortionType.HORIZONTAL_INTERLACED:
            offset_x = -raw_offset if y % 2 == 0 else raw_offset
            src_y = y % src_h
        else:  # VERTICAL
            offset_x = 0
            src_y = int((raw_offset + y * compression) % src_h)

        for x in range(SNES_WIDTH):
            if effect.type in (DistortionType.HORIZONTAL, DistortionType.HORIZONTAL_INTERLACED):
                sx = (x + offset_x) % src_w
            else:
                sx = x % src_w

            dst[y, x] = source[src_y, sx]

    return (dst * alpha).clip(0, 255)


def compute_frame_fast(
    source: np.ndarray,
    tick: int,
    effect: DistortionEffect,
    alpha: float,
    letterbox: int = 0,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
) -> np.ndarray:
    """Vectorized version of compute_frame for performance."""
    t2 = tick * 2
    amplitude = C1 * (effect.amplitude + effect.amplitude_accel * t2)
    frequency = C2 * (effect.frequency + effect.frequency_accel * t2)
    compression = 1.0 + (effect.compression + effect.compression_accel * t2) / 256.0
    speed = C3 * effect.speed * tick

    src_h, src_w = source.shape[:2]
    ys = np.arange(SNES_HEIGHT)
    xs = np.arange(SNES_WIDTH)

    # Background scroll: constant pan over time
    sx = int(scroll_x * tick) % src_w
    sy = int(scroll_y * tick) % src_h

    raw_offsets = np.round(amplitude * np.sin(frequency * ys + speed)).astype(np.int32)

    if effect.type == DistortionType.HORIZONTAL:
        offset_x = raw_offsets
        src_ys = (ys + sy) % src_h
    elif effect.type == DistortionType.HORIZONTAL_INTERLACED:
        signs = np.where(ys % 2 == 0, -1, 1)
        offset_x = raw_offsets * signs
        src_ys = (ys + sy) % src_h
    else:  # VERTICAL
        offset_x = np.zeros(SNES_HEIGHT, dtype=np.int32)
        src_ys = ((raw_offsets + ((ys + sy) * compression).astype(np.int32)) % src_h).astype(np.int32)

    # src_x[y, x] = (x + offset_x[y] + scroll_x_offset) % src_w
    src_x_grid = (xs[np.newaxis, :] + offset_x[:, np.newaxis] + sx) % src_w
    src_y_grid = np.broadcast_to(src_ys[:, np.newaxis], (SNES_HEIGHT, SNES_WIDTH))

    dst = source[src_y_grid, src_x_grid].astype(np.float64) * alpha

    if letterbox > 0:
        dst[:letterbox] = 0
        dst[SNES_HEIGHT - letterbox:] = 0

    return dst.clip(0, 255)


def cycle_palette(base_img: np.ndarray, palette: list, shift: int) -> np.ndarray:
    """Cycle colors in the image by shifting the palette mapping."""
    if shift == 0:
        return base_img
    n = len(palette)
    result = np.copy(base_img)
    for i, color in enumerate(palette):
        new_color = palette[(i + shift) % n]
        mask = np.all(base_img == color, axis=2)
        result[mask] = new_color
    return result


def cycle_palette_fast(base_img: np.ndarray, palette: list, shift: int) -> np.ndarray:
    """Faster palette cycling using a lookup approach."""
    if shift == 0:
        return base_img.copy()
    n = len(palette)
    # Build a color remap LUT: for each palette entry, map to shifted entry
    old_palette = np.array(palette, dtype=np.uint8)
    new_palette = np.array([palette[(i + shift) % n] for i in range(n)], dtype=np.uint8)

    result = base_img.copy()
    for i in range(n):
        mask = np.all(base_img == old_palette[i], axis=2)
        result[mask] = new_palette[i]
    return result


# --- Presets ---

PRESETS = {
    "fire": [
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL,
                frequency=2048,
                amplitude=4096,
                compression=0,
                frequency_accel=0,
                amplitude_accel=0,
                speed=1200,
                compression_accel=0,
            ),
            palette=[
                (10, 0, 0), (40, 0, 0), (80, 10, 0), (130, 30, 0),
                (180, 50, 0), (220, 80, 0), (255, 120, 0), (255, 160, 20),
                (255, 200, 60), (255, 230, 100), (255, 200, 60), (255, 160, 20),
                (220, 80, 0), (180, 50, 0), (130, 30, 0), (80, 10, 0),
            ],
            palette_cycle_speed=2,
            alpha=1.0,
            texture_func="horizontal_stripes",
            scroll_x=0.0,
            scroll_y=-1.0,
        ),
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL_INTERLACED,
                frequency=4096,
                amplitude=2048,
                compression=0,
                frequency_accel=0,
                amplitude_accel=0,
                speed=1440,
                compression_accel=0,
            ),
            palette=[
                (0, 0, 0), (20, 0, 0), (60, 0, 0), (100, 20, 0),
                (140, 40, 0), (100, 20, 0), (60, 0, 0), (20, 0, 0),
            ],
            palette_cycle_speed=4,
            alpha=0.5,
            texture_func="plasma",
            scroll_x=1.0,
            scroll_y=-1.0,
        ),
    ],
    "ocean": [
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL,
                frequency=1024,
                amplitude=3072,
                compression=0,
                frequency_accel=0,
                amplitude_accel=0,
                speed=720,
                compression_accel=0,
            ),
            palette=[
                (0, 0, 20), (0, 10, 50), (0, 20, 80), (0, 40, 120),
                (0, 60, 160), (0, 80, 200), (20, 100, 220), (40, 130, 240),
                (60, 150, 255), (40, 130, 240), (20, 100, 220), (0, 80, 200),
                (0, 60, 160), (0, 40, 120), (0, 20, 80), (0, 10, 50),
            ],
            palette_cycle_speed=2,
            alpha=1.0,
            texture_func="horizontal_stripes",
            scroll_x=1.0,
            scroll_y=1.0,
        ),
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.VERTICAL,
                frequency=2048,
                amplitude=1536,
                compression=256,
                frequency_accel=0,
                amplitude_accel=0,
                speed=480,
                compression_accel=0,
            ),
            palette=[
                (0, 0, 40), (0, 20, 80), (0, 50, 130), (0, 80, 180),
                (10, 100, 200), (0, 80, 180), (0, 50, 130), (0, 20, 80),
            ],
            palette_cycle_speed=4,
            alpha=0.4,
            texture_func="concentric",
            scroll_x=-1.0,
            scroll_y=1.0,
        ),
    ],
    "cosmic": [
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.VERTICAL,
                frequency=3072,
                amplitude=2048,
                compression=512,
                frequency_accel=8,
                amplitude_accel=0,
                speed=1024,
                compression_accel=4,
            ),
            palette=[
                (0, 0, 0), (20, 0, 40), (40, 0, 80), (80, 0, 160),
                (120, 0, 200), (160, 0, 255), (200, 50, 255), (255, 100, 255),
                (200, 50, 255), (160, 0, 255), (120, 0, 200), (80, 0, 160),
                (40, 0, 80), (20, 0, 40),
            ],
            palette_cycle_speed=2,
            alpha=1.0,
            texture_func="plasma",
            scroll_x=0.5,
            scroll_y=0.5,
        ),
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL_INTERLACED,
                frequency=6144,
                amplitude=1024,
                compression=0,
                frequency_accel=0,
                amplitude_accel=4,
                speed=2048,
                compression_accel=0,
            ),
            palette=[
                (0, 0, 20), (0, 0, 60), (20, 0, 100), (50, 20, 140),
                (20, 0, 100), (0, 0, 60),
            ],
            palette_cycle_speed=1,
            alpha=0.35,
            texture_func="diagonal_stripes",
            scroll_x=-0.5,
            scroll_y=0.5,
        ),
    ],
    "acid": [
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL,
                frequency=5120,
                amplitude=3072,
                compression=0,
                frequency_accel=16,
                amplitude_accel=8,
                speed=2048,
                compression_accel=0,
            ),
            palette=[
                (0, 80, 0), (0, 120, 0), (0, 180, 0), (0, 220, 40),
                (0, 255, 80), (40, 255, 120), (80, 255, 160), (120, 255, 200),
                (80, 255, 160), (40, 255, 120), (0, 255, 80), (0, 220, 40),
                (0, 180, 0), (0, 120, 0),
            ],
            palette_cycle_speed=1,
            alpha=1.0,
            texture_func="checkerboard",
            scroll_x=0.5,
            scroll_y=0.5,
        ),
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.VERTICAL,
                frequency=2048,
                amplitude=4096,
                compression=384,
                frequency_accel=0,
                amplitude_accel=0,
                speed=1536,
                compression_accel=8,
            ),
            palette=[
                (0, 40, 0), (0, 80, 20), (20, 120, 40), (40, 160, 60),
                (20, 120, 40), (0, 80, 20),
            ],
            palette_cycle_speed=2,
            alpha=0.45,
            texture_func="concentric",
            scroll_x=-0.5,
            scroll_y=0.5,
        ),
    ],
    "cave": [
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL_INTERLACED,
                frequency=3072,
                amplitude=2048,
                compression=0,
                frequency_accel=0,
                amplitude_accel=0,
                speed=480,
                compression_accel=0,
            ),
            palette=[
                (20, 10, 5), (40, 20, 10), (60, 30, 15), (80, 40, 20),
                (100, 50, 25), (120, 60, 30), (100, 50, 25), (80, 40, 20),
            ],
            palette_cycle_speed=4,
            alpha=1.0,
            texture_func="diagonal_stripes",
            scroll_x=1.0,
            scroll_y=1.0,
        ),
        LayerConfig(
            effect=DistortionEffect(
                type=DistortionType.HORIZONTAL,
                frequency=1536,
                amplitude=1024,
                compression=0,
                frequency_accel=0,
                amplitude_accel=0,
                speed=360,
                compression_accel=0,
            ),
            palette=[
                (10, 5, 0), (30, 15, 5), (50, 25, 10), (70, 35, 15),
                (90, 45, 20), (70, 35, 15), (50, 25, 10), (30, 15, 5),
            ],
            palette_cycle_speed=4,
            alpha=0.5,
            texture_func="horizontal_stripes",
            scroll_x=-1.0,
            scroll_y=1.0,
        ),
    ],
}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _gcd(a, b)


def compute_loop_frames(preset_name: str) -> int | None:
    """Compute the number of frames for a perfect loop, or None if impossible."""
    layers = PRESETS[preset_name]

    for lc in layers:
        e = lc.effect
        if e.frequency_accel != 0 or e.amplitude_accel != 0 or e.compression_accel != 0:
            return None

    period = 1
    for lc in layers:
        e = lc.effect

        # Distortion phase period: C3 * speed * T = 2π*k → T = 120k/speed
        # Smallest integer T: T = 120 / gcd(120, speed)
        if e.speed != 0:
            dist_period = 120 // _gcd(120, abs(e.speed))
            period = _lcm(period, dist_period)

        # Palette cycle period
        if lc.palette_cycle_speed > 0:
            pal_period = lc.palette_cycle_speed * len(lc.palette)
            period = _lcm(period, pal_period)

        # Scroll periods: scroll * T ≡ 0 (mod 256)
        # T = 256 / scroll, but scroll may be fractional.
        # Express scroll as a fraction p/q, then T = 256*q/p
        for scroll in (lc.scroll_x, lc.scroll_y):
            if scroll == 0:
                continue
            from fractions import Fraction
            frac = Fraction(abs(scroll)).limit_denominator(1000)
            scroll_period = int(256 * frac.denominator / frac.numerator)
            if scroll_period > 0:
                period = _lcm(period, scroll_period)

    return period


def generate_frames(preset_name: str, duration: float):
    """Generator yielding (frame_index, frame_array) tuples."""
    if preset_name not in PRESETS:
        print(f"Unknown preset '{preset_name}'. Available: {', '.join(PRESETS.keys())}")
        sys.exit(1)

    layers_config = PRESETS[preset_name]
    total_frames = int(duration * FPS)

    # Pre-generate base textures for each layer
    base_textures = []
    for lc in layers_config:
        gen_func = TEXTURE_GENERATORS[lc.texture_func]
        base_textures.append(gen_func(lc.palette))

    for frame_idx in range(total_frames):
        composite = np.zeros((SNES_HEIGHT, SNES_WIDTH, 3), dtype=np.float64)

        for layer_idx, lc in enumerate(layers_config):
            if lc.palette_cycle_speed > 0:
                shift = frame_idx // lc.palette_cycle_speed
                src = cycle_palette_fast(base_textures[layer_idx], lc.palette, shift % len(lc.palette))
            else:
                src = base_textures[layer_idx]

            frame_data = compute_frame_fast(src, frame_idx, lc.effect, lc.alpha,
                                            scroll_x=lc.scroll_x, scroll_y=lc.scroll_y)

            if layer_idx == 0:
                composite = frame_data
            else:
                composite += frame_data

        yield frame_idx, composite.clip(0, 255).astype(np.uint8)


def render_video(preset_name: str, duration: float, output_path: str, scale: int = 3):
    """Render a full video of the battle background animation."""
    total_frames = int(duration * FPS)
    out_w, out_h = SNES_WIDTH * scale, SNES_HEIGHT * scale

    print(f"Rendering '{preset_name}': {out_w}x{out_h} @ {FPS}fps, {duration}s ({total_frames} frames)")
    print("Generating textures...")

    if HAS_IMAGEIO:
        _render_with_imageio(preset_name, duration, output_path, scale)
    else:
        _render_with_ffmpeg_cli(preset_name, duration, output_path, scale)

    print(f"Done! Output: {output_path}")


def _render_with_imageio(preset_name: str, duration: float, output_path: str, scale: int):
    """Render using imageio (includes bundled ffmpeg)."""
    total_frames = int(duration * FPS)
    out_w, out_h = SNES_WIDTH * scale, SNES_HEIGHT * scale

    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=FPS, pixel_format="yuv420p")

    for frame_idx, frame in generate_frames(preset_name, duration):
        if frame_idx % FPS == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx * 100 // total_frames}%)")
        # Nearest-neighbor upscale
        scaled = frame.repeat(scale, axis=0).repeat(scale, axis=1)
        writer.write_frame(scaled)

    writer.close()


def _render_with_ffmpeg_cli(preset_name: str, duration: float, output_path: str, scale: int):
    """Render by piping raw frames to system ffmpeg."""
    total_frames = int(duration * FPS)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{SNES_WIDTH}x{SNES_HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={SNES_WIDTH * scale}:{SNES_HEIGHT * scale}:flags=neighbor",
        output_path,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for frame_idx, frame in generate_frames(preset_name, duration):
        if frame_idx % FPS == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx * 100 // total_frames}%)")
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        print(f"ffmpeg error:\n{stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EarthBound battle background generator")
    parser.add_argument("--preset", default="cosmic", choices=list(PRESETS.keys()),
                        help="Background preset to render")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds (default: 10)")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: <preset>.mp4)")
    parser.add_argument("--loop", action="store_true",
                        help="Auto-compute duration for a perfect loop")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name in PRESETS:
            frames = compute_loop_frames(name)
            if frames:
                print(f"  {name:10s} (loop: {frames} frames / {frames / FPS:.2f}s)")
            else:
                print(f"  {name:10s} (no perfect loop — has acceleration)")
        sys.exit(0)

    duration = args.duration
    if args.loop:
        frames = compute_loop_frames(args.preset)
        if frames is None:
            print(f"Warning: '{args.preset}' uses acceleration and cannot loop perfectly.")
            print(f"Using --duration {args.duration}s instead.")
        else:
            duration = frames / FPS
            print(f"Loop point: {frames} frames = {duration:.2f}s")

    output = args.output or f"{args.preset}.mp4"
    render_video(args.preset, duration, output)


if __name__ == "__main__":
    main()
