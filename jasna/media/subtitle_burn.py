"""Burn ASS/SSA subtitles into frames on the GPU during the single encode pass.

PyAV's bundled FFmpeg has no libass, so the text is rasterized by the external
``ffmpeg`` binary (the one the release bundles next to ``ffprobe``) and only
the rendered pixels cross the process boundary. The subtitle file is rendered
twice inside one filtergraph at the output frame rate, over black and over
white. For a pixel with subtitle coverage ``a`` and premultiplied colour ``C``:

    black render  B = C
    white render  W = 255 * (1 - a) + C

so ``W - B`` is the exact remaining transparency and the video frame is
composited as ``out = video * (W - B) / 255 + B``. The ``ass`` filter's own
``alpha=1`` mode cannot replace this: FFmpeg's drawutils blend the alpha plane
with the layer alpha as both source and weight, so the stacked fill, outline
and shadow layers leave a quadratic, not accumulated, coverage there.

Both renders come back as one vertically stacked rawvideo stream, and a reader
thread stays a few frames ahead so the encoder worker only pays for the upload
and the composite, and only on frames that carry text.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

SUBTITLE_SUFFIXES = (".ass", ".ssa")
BURN_SUBTITLES_AUTO = "auto"

_STOP = object()


def resolve_burn_subtitles(spec: str | Path | None, input_video: Path) -> Path | None:
    """Turn the user-facing setting into the subtitle file for one video.

    ``auto`` looks for a sidecar next to the input (``video.ass``, then
    ``video.ssa``) and yields ``None`` when there is none; anything else is a
    path that must exist.
    """
    value = str(spec or "").strip()
    if not value:
        return None
    if value.lower() == BURN_SUBTITLES_AUTO:
        for suffix in SUBTITLE_SUFFIXES:
            candidate = Path(input_video).with_suffix(suffix)
            if candidate.is_file():
                return candidate
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Subtitle file not found: {path}")
    return path


def _escape_filter_option(value: str) -> str:
    # ffmpeg parses a filtergraph twice: the option parser treats ``:`` and
    # ``\`` specially, then the graph parser strips one level of quoting.
    # Windows paths become forward-slash paths so the drive colon is the only
    # thing left to escape.
    if os.name == "nt":
        value = value.replace("\\", "/")
    escaped = value.replace("\\", "\\\\").replace(":", "\\:")
    return "'" + escaped.replace("'", "'\\''") + "'"


def build_subtitle_render_command(
    subtitle_path: str | Path,
    *,
    width: int,
    height: int,
    fps: Fraction,
    fonts_dir: str | Path | None = None,
    ffmpeg_path: str | None = None,
) -> list[str]:
    fps = Fraction(fps)
    ass_options = f"filename={_escape_filter_option(str(Path(subtitle_path).resolve()))}"
    if fonts_dir:
        ass_options += f":fontsdir={_escape_filter_option(str(Path(fonts_dir).resolve()))}"
    source = f"s={width}x{height}:r={fps.numerator}/{fps.denominator}"
    graph = (
        f"color=c=black:{source},format=rgb24,ass={ass_options}[b];"
        f"color=c=white:{source},format=rgb24,ass={ass_options}[w];"
        "[b][w]vstack"
    )
    return [
        ffmpeg_path or resolve_executable("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel", "error",
        "-filter_complex", graph,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]


def composite_subtitle_overlay(
    frame_chw: torch.Tensor, render_hwc: torch.Tensor
) -> torch.Tensor:
    """Blend a stacked black/white render onto a uint8 ``(3, H, W)`` frame."""
    if frame_chw.ndim != 3 or frame_chw.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) RGB tensor, got {tuple(frame_chw.shape)}")
    height = frame_chw.shape[1]
    if tuple(render_hwc.shape) != (2 * height, frame_chw.shape[2], 3):
        raise ValueError(
            f"Subtitle render {tuple(render_hwc.shape)} does not match frame "
            f"{tuple(frame_chw.shape)}"
        )
    black = render_hwc[:height].permute(2, 0, 1).to(torch.float32)
    white = render_hwc[height:].permute(2, 0, 1).to(torch.float32)
    transparency = white.sub_(black).div_(255.0)
    out = frame_chw.to(torch.float32).mul_(transparency).add_(black)
    return out.round_().clamp_(0.0, 255.0).to(frame_chw.dtype)


def _render_has_text(render: np.ndarray, height: int) -> bool:
    # A pixel with any coverage darkens the white render, unless the subtitle
    # itself is pure white, in which case it lights up the black render.
    return bool(render[height:].min() != 255) or bool(render[:height].any())


class AssSubtitleBurner:
    """Rasterizes one subtitle file with ffmpeg and composites it onto frames.

    ``composite`` must be called with non-decreasing timestamps: the render
    stream is consumed forward only, and a frame that maps to an earlier render
    than the last one reuses the last render.
    """

    def __init__(
        self,
        subtitle_path: str | Path,
        *,
        width: int,
        height: int,
        fps: Fraction,
        device: torch.device,
        fonts_dir: str | Path | None = None,
        ffmpeg_path: str | None = None,
        prefetch: int = 4,
    ):
        self.subtitle_path = Path(subtitle_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = Fraction(fps)
        self.device = torch.device(device)
        self._frame_shape = (2 * self.height, self.width, 3)
        self._frame_bytes = 2 * self.height * self.width * 3
        self._command = build_subtitle_render_command(
            self.subtitle_path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            fonts_dir=fonts_dir,
            ffmpeg_path=ffmpeg_path,
        )
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(prefetch)))
        self._current_index = -1
        self._current: np.ndarray | None = None
        self._current_gpu: torch.Tensor | None = None
        self._error: Exception | None = None
        self._closed = False
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_no_window_kwargs(),
        )
        self._reader = threading.Thread(
            target=self._read_renders, name="AssSubtitleReader", daemon=True
        )
        self._reader.start()

    def _read_renders(self) -> None:
        stdout = self._process.stdout
        index = 0
        scratch = np.empty(self._frame_shape, dtype=np.uint8)
        try:
            while True:
                view = memoryview(scratch).cast("B")
                filled = 0
                while filled < self._frame_bytes:
                    n = stdout.readinto(view[filled:])
                    if not n:
                        break
                    filled += n
                if filled < self._frame_bytes:
                    self._queue.put(self._eof_error(index))
                    return
                if _render_has_text(scratch, self.height):
                    self._queue.put((index, scratch))
                    scratch = np.empty(self._frame_shape, dtype=np.uint8)
                else:
                    self._queue.put((index, None))
                index += 1
        except Exception as exc:  # pragma: no cover - defensive
            self._queue.put(exc)

    def _eof_error(self, rendered: int) -> Exception:
        # The colour source is unbounded, so ffmpeg only stops on its own when
        # something went wrong (missing filter, unreadable file, font setup).
        if self._closed:
            return RuntimeError("Subtitle renderer closed")
        try:
            stderr = self._process.stderr.read().decode(errors="replace").strip()
        except Exception:
            stderr = ""
        code = self._process.wait()
        detail = stderr or f"exit code {code}"
        hint = ""
        if "ass" in detail and ("No such filter" in detail or "Unable to find" in detail):
            hint = " (this ffmpeg build has no libass; use a full build)"
        return RuntimeError(
            f"ffmpeg stopped rendering subtitles after {rendered} frame(s): {detail}{hint}"
        )

    def _advance_to(self, target_index: int) -> None:
        while self._current_index < target_index:
            if self._error is not None:
                raise self._error
            item = self._queue.get()
            if isinstance(item, Exception):
                self._error = item
                raise item
            self._current_index, self._current = item
            self._current_gpu = None

    def composite(self, frame_chw: torch.Tensor, seconds: float) -> torch.Tensor:
        """Return ``frame_chw`` with the subtitles for ``seconds`` burned in.

        The frame itself is returned when nothing is displayed at that time.
        """
        if self._closed:
            raise RuntimeError("Subtitle renderer closed")
        target = max(0, int(round(float(seconds) * self.fps)))
        self._advance_to(target)
        if self._current is None:
            return frame_chw
        if self._current_gpu is None:
            self._current_gpu = torch.from_numpy(self._current).to(self.device)
        return composite_subtitle_overlay(frame_chw, self._current_gpu)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None:
            process.kill()
        process.wait()
        # Killing ffmpeg ends the reader with EOF; keep draining so it is never
        # parked on a full queue while we wait for it.
        deadline = 50
        while self._reader.is_alive() and deadline > 0:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._reader.join(timeout=0.1)
            deadline -= 1
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
