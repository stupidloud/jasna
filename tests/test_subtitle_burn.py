"""Subtitle burn-in: sidecar resolution, ffmpeg command, compositing, and the render reader."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

import jasna.media.subtitle_burn as subtitle_burn
from jasna.media.subtitle_burn import (
    AssSubtitleBurner,
    build_subtitle_render_command,
    composite_subtitle_overlay,
    resolve_burn_subtitles,
)


# ---------------------------------------------------------------- resolution


def test_resolve_empty_spec_is_none(tmp_path: Path) -> None:
    assert resolve_burn_subtitles(None, tmp_path / "in.mp4") is None
    assert resolve_burn_subtitles("   ", tmp_path / "in.mp4") is None


def test_resolve_auto_prefers_ass_sidecar_then_ssa(tmp_path: Path) -> None:
    video = tmp_path / "movie.mkv"
    video.touch()
    assert resolve_burn_subtitles("auto", video) is None

    ssa = tmp_path / "movie.ssa"
    ssa.touch()
    assert resolve_burn_subtitles("AUTO", video) == ssa

    ass = tmp_path / "movie.ass"
    ass.touch()
    assert resolve_burn_subtitles("auto", video) == ass


def test_resolve_explicit_path_must_exist(tmp_path: Path) -> None:
    sub = tmp_path / "subs.ass"
    with pytest.raises(FileNotFoundError):
        resolve_burn_subtitles(str(sub), tmp_path / "in.mp4")
    sub.touch()
    assert resolve_burn_subtitles(str(sub), tmp_path / "in.mp4") == sub


# ------------------------------------------------------------------ command


def test_render_command_escapes_paths_and_stacks_black_over_white(tmp_path: Path) -> None:
    sub_dir = tmp_path / "sub dir"
    sub_dir.mkdir()
    sub = sub_dir / "it's.ass"
    sub.touch()
    fonts = tmp_path / "fonts"
    fonts.mkdir()

    cmd = build_subtitle_render_command(
        sub,
        width=1920,
        height=1080,
        fps=Fraction(30000, 1001),
        fonts_dir=fonts,
        ffmpeg_path="ffmpeg-bin",
    )

    assert cmd[0] == "ffmpeg-bin"
    assert cmd[-5:] == ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("color=c=black:s=1920x1080:r=30000/1001,format=rgb24,ass=") == 1
    assert graph.count("color=c=white:s=1920x1080:r=30000/1001,format=rgb24,ass=") == 1
    assert graph.endswith("[b][w]vstack")
    # Drive colons are escaped for the option parser and the whole value is
    # quoted for the graph parser; the apostrophe survives the quoting.
    resolved = str(sub.resolve())
    if os.name == "nt":
        resolved = resolved.replace("\\", "/")
    assert "filename='" + resolved.replace(":", "\\:").replace("'", "'\\''") + "'" in graph
    assert ":fontsdir='" in graph


def test_render_command_defaults_to_resolved_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    sub = tmp_path / "s.ass"
    sub.touch()
    monkeypatch.setattr(subtitle_burn, "resolve_executable", lambda name: f"/opt/{name}")
    cmd = build_subtitle_render_command(sub, width=16, height=8, fps=Fraction(25))
    assert cmd[0] == "/opt/ffmpeg"
    assert "fontsdir" not in cmd[cmd.index("-filter_complex") + 1]


# --------------------------------------------------------------- compositing


def _stacked_render(height: int, width: int, colour, alpha: float) -> torch.Tensor:
    """Black and white renders of one flat premultiplied colour at coverage ``alpha``."""
    colour = np.asarray(colour, dtype=np.float32)
    black = np.broadcast_to(colour * alpha, (height, width, 3))
    white = np.broadcast_to(255.0 * (1.0 - alpha) + colour * alpha, (height, width, 3))
    stacked = np.concatenate([black, white], axis=0)
    return torch.from_numpy(np.round(stacked).astype(np.uint8))


def test_composite_without_coverage_is_identity() -> None:
    frame = torch.randint(0, 256, (3, 6, 8), dtype=torch.uint8)
    render = _stacked_render(6, 8, (0, 0, 0), 0.0)
    assert torch.equal(composite_subtitle_overlay(frame, render), frame)


def test_composite_opaque_colour_replaces_frame() -> None:
    frame = torch.randint(0, 256, (3, 6, 8), dtype=torch.uint8)
    render = _stacked_render(6, 8, (200, 30, 90), 1.0)
    out = composite_subtitle_overlay(frame, render)
    assert out.dtype == torch.uint8
    assert torch.equal(out[0], torch.full((6, 8), 200, dtype=torch.uint8))
    assert torch.equal(out[1], torch.full((6, 8), 30, dtype=torch.uint8))
    assert torch.equal(out[2], torch.full((6, 8), 90, dtype=torch.uint8))


def test_composite_half_coverage_matches_reference_blend() -> None:
    frame = torch.randint(0, 256, (3, 6, 8), dtype=torch.uint8)
    render = _stacked_render(6, 8, (255, 255, 255), 0.5)
    out = composite_subtitle_overlay(frame, render).to(torch.float32)
    expected = frame.to(torch.float32) * 0.5 + 127.5
    assert (out - expected).abs().max() <= 1.0


def test_composite_rejects_mismatched_shapes() -> None:
    frame = torch.zeros((3, 6, 8), dtype=torch.uint8)
    with pytest.raises(ValueError):
        composite_subtitle_overlay(frame, torch.zeros((6, 8, 3), dtype=torch.uint8))
    with pytest.raises(ValueError):
        composite_subtitle_overlay(torch.zeros((6, 8, 3), dtype=torch.uint8), torch.zeros((12, 8, 3), dtype=torch.uint8))


# ------------------------------------------------------------ render reader


_FAKE_FFMPEG = textwrap.dedent(
    """
    import sys, time
    height, width, count, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out = sys.stdout.buffer
    for index in range(count):
        if mode == "text" and index % 2 == 1:
            # Frame index n carries the flat colour (n, 0, 0) at full coverage.
            black = bytes([index, 0, 0]) * (height * width)
            white = bytes([index, 0, 0]) * (height * width)
        else:
            black = bytes([0, 0, 0]) * (height * width)
            white = bytes([255, 255, 255]) * (height * width)
        out.write(black + white)
    out.flush()
    if mode == "fail":
        sys.stderr.write("No such filter: 'ass'\\n")
        sys.exit(1)
    # Stay alive like ffmpeg's unbounded colour source would.
    time.sleep(30)
    """
)


def _fake_burner(monkeypatch, tmp_path: Path, *, count: int, mode: str, **kwargs) -> AssSubtitleBurner:
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(_FAKE_FFMPEG, encoding="utf-8")
    sub = tmp_path / "s.ass"
    sub.touch()

    def fake_command(subtitle_path, *, width, height, fps, fonts_dir=None, ffmpeg_path=None):
        return [sys.executable, str(script), str(height), str(width), str(count), mode]

    monkeypatch.setattr(subtitle_burn, "build_subtitle_render_command", fake_command)
    return AssSubtitleBurner(sub, width=4, height=2, fps=Fraction(10), device=torch.device("cpu"), **kwargs)


def test_burner_maps_seconds_to_render_index_and_skips_empty_frames(monkeypatch, tmp_path: Path) -> None:
    burner = _fake_burner(monkeypatch, tmp_path, count=8, mode="text", prefetch=2)
    try:
        frame = torch.full((3, 2, 4), 77, dtype=torch.uint8)
        # 0.0 s -> render 0 (empty): the frame object itself comes back.
        assert burner.composite(frame, 0.0) is frame
        # 0.3 s -> render 3 (text): flat colour (3, 0, 0).
        out = burner.composite(frame, 0.3)
        assert out[0].unique().tolist() == [3] and out[1].unique().tolist() == [0]
        # Earlier timestamps reuse the current render instead of seeking back.
        assert torch.equal(burner.composite(frame, 0.1), out)
        # 0.62 s rounds to render 6 (empty), skipping render 4 and 5.
        assert burner.composite(frame, 0.62) is frame
        assert burner.composite(frame, 0.7)[0].unique().tolist() == [7]
    finally:
        burner.close()
    assert burner._process.poll() is not None
    assert not burner._reader.is_alive()


def test_burner_surfaces_ffmpeg_failure_with_stderr(monkeypatch, tmp_path: Path) -> None:
    burner = _fake_burner(monkeypatch, tmp_path, count=2, mode="fail")
    try:
        frame = torch.zeros((3, 2, 4), dtype=torch.uint8)
        assert burner.composite(frame, 0.1) is frame
        with pytest.raises(RuntimeError, match="No such filter") as exc:
            burner.composite(frame, 1.0)
        assert "libass" in str(exc.value)
        with pytest.raises(RuntimeError):
            burner.composite(frame, 2.0)
    finally:
        burner.close()


def test_burner_close_is_idempotent_and_rejects_further_use(monkeypatch, tmp_path: Path) -> None:
    burner = _fake_burner(monkeypatch, tmp_path, count=4, mode="text")
    burner.close()
    burner.close()
    with pytest.raises(RuntimeError):
        burner.composite(torch.zeros((3, 2, 4), dtype=torch.uint8), 0.0)


# --------------------------------------------------------- real ffmpeg check


_ASS = textwrap.dedent(
    """
    [Script Info]
    ScriptType: v4.00+
    PlayResX: 320
    PlayResY: 180

    [V4+ Styles]
    Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    Style: Default,Arial,32,&H4000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,2,10,10,10,1

    [Events]
    Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    Dialogue: 0,0:00:00.40,0:00:01.20,Default,,0,0,0,,Hello {\\c&H00FF00&}world
    Dialogue: 1,0:00:00.80,0:00:01.60,Default,,0,0,0,,{\\an8\\alpha&H60&}Top layer
    """
)


def _ffmpeg_with_ass() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    try:
        filters = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return None
    return ffmpeg if " ass " in filters else None


@pytest.mark.skipif(_ffmpeg_with_ass() is None, reason="needs an ffmpeg with libass on PATH")
def test_burner_matches_ffmpeg_direct_burn_in(tmp_path: Path) -> None:
    ffmpeg = _ffmpeg_with_ass()
    width, height, fps, seconds = 320, 180, 25, 2
    sub = tmp_path / "ref.ass"
    sub.write_text(_ASS, encoding="utf-8")
    grey = 0x70

    reference = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x{grey:02x}{grey:02x}{grey:02x}:s={width}x{height}:r={fps}:d={seconds}",
            "-vf", f"format=rgb24,ass=filename='{str(sub.resolve()).replace(chr(92), '/').replace(':', chr(92) + ':')}'",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    reference = np.frombuffer(reference, np.uint8).reshape(-1, height, width, 3)

    burner = AssSubtitleBurner(
        sub, width=width, height=height, fps=Fraction(fps), device=torch.device("cpu"), ffmpeg_path=ffmpeg
    )
    try:
        frame = torch.full((3, height, width), grey, dtype=torch.uint8)
        worst = 0
        off_by_two = 0
        text_frames = 0
        for index in range(reference.shape[0]):
            out = burner.composite(frame, index / fps)
            if out is not frame:
                text_frames += 1
            diff = (out.permute(1, 2, 0).to(torch.int16) - torch.from_numpy(reference[index].astype(np.int16))).abs()
            worst = max(worst, int(diff.max()))
            off_by_two += int((diff >= 2).sum())
    finally:
        burner.close()

    # Both renders are 8-bit, so recovering the coverage from their difference
    # can land one step further from ffmpeg's own rounding on a few edge pixels.
    assert text_frames > 0
    assert worst <= 2
    assert off_by_two < reference.shape[0] * 50
