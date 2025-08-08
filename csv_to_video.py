#!/usr/bin/env python3
"""
CSV→Video renderer

Spec:
- Input: CSV from stdin with header: time[ms], pitch, length[ms] (velocity ignored)
- Behavior:
  - For each row, display `pitch` centered on screen starting at time[ms] for length[ms]
  - Texts may overlap; each line becomes an independent clip
  - Fade-in/out applied per text for `--fade-ms` (0 disables fades)
  - Video size: `--size` (default 480x270); background color `--bg` (default #000000)
  - Font: `--font` name/path (system default if omitted); `--font-size` (default ≈20% of height)
  - Text color: `--text-color` (default #FFFFFF)
  - Tail margin after last event: `--tail-ms` (default 500ms)
  - Optional BGM: `--bgm` path, `--bgm-volume` (default 1.0). Start aligned at 0s
  - Total duration = max(events_end + tail, bgm_duration if present)
- Output: MP4 (H.264 yuv420p). With AAC if BGM is provided; silent otherwise
- Dependencies: moviepy, Pillow, ffmpeg installed in system
- Notes:
  - Text rendering via Pillow; includes ascent/descent offset compensation and generous padding to avoid cropping
  - Designed for thousands of events; very large inputs may increase render time

Extended behavior (no overlap mode implemented):
- Only one text is visible at any time
- If a new group starts, the previous text is cut off immediately (even if its length remains)
- If multiple notes share the same time, they are displayed side-by-side as "Ab1, C1" sorted by keyboard order
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageColor
from moviepy.editor import ImageClip, ColorClip, CompositeVideoClip, AudioFileClip
from moviepy.video.fx import all as vfx
import numpy as np


@dataclass
class NoteEvent:
    start_s: float
    duration_s: float
    pitch: str


def parse_size(text: str) -> Tuple[int, int]:
    try:
        w_str, h_str = text.lower().split('x')
        w = int(w_str)
        h = int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size '{text}', expected WIDTHxHEIGHT, e.g. 480x270")


def parse_color(text: str) -> Tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(text)
        if len(rgb) == 4:
            rgb = rgb[:3]
        return tuple(int(c) for c in rgb)  # type: ignore[return-value]
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid color '{text}'")


def resolve_font(font_name_or_path: Optional[str], font_size: int) -> ImageFont.ImageFont:
    if not font_name_or_path:
        try:
            return ImageFont.truetype("Arial.ttf", font_size)  # type: ignore[return-value]
        except Exception:
            return ImageFont.load_default()

    # Try as a direct file path first
    try:
        return ImageFont.truetype(font_name_or_path, font_size)  # type: ignore[return-value]
    except Exception:
        pass

    # Try resolve by scanning common macOS/Linux font directories
    import os
    import glob

    candidates: List[str] = []
    font_dirs = [
        os.path.expanduser("~/Library/Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]
    patterns = ["*.ttf", "*.otf", "*.ttc"]
    key = font_name_or_path.lower()

    for d in font_dirs:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            for path in glob.glob(f"{d}/{p}"):
                filename = os.path.basename(path).lower()
                if key in filename:
                    candidates.append(path)

    if candidates:
        try:
            return ImageFont.truetype(candidates[0], font_size)  # type: ignore[return-value]
        except Exception:
            pass

    # Fallbacks
    try:
        return ImageFont.truetype("Arial.ttf", font_size)  # type: ignore[return-value]
    except Exception:
        return ImageFont.load_default()


def render_text_image(text: str, font: ImageFont.ImageFont, color: Tuple[int, int, int]) -> Image.Image:
    dummy_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)
    offset_x = 0
    offset_y = 0
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        left, top, right, bottom = bbox
        text_w = right - left
        text_h = bottom - top
        offset_x = -left
        offset_y = -top
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
        offset_x = 0
        offset_y = 0

    base_size = getattr(font, 'size', max(text_w, text_h))
    pad_x = max(8, int(base_size * 0.15))
    pad_y = max(8, int(base_size * 0.30))

    img = Image.new("RGBA", (text_w + 2 * pad_x, text_h + 2 * pad_y), (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(img)
    draw2.text((pad_x + offset_x, pad_y + offset_y), text, font=font, fill=(color[0], color[1], color[2], 255))
    return img


def read_events_from_stdin() -> List[NoteEvent]:
    reader = csv.DictReader(sys.stdin)
    required_columns = {"time[ms]", "pitch", "length[ms]"}
    if reader.fieldnames is None or not required_columns.issubset(set(fn.strip() for fn in reader.fieldnames)):
        raise SystemExit("CSV header must include: time[ms], pitch, length[ms]")

    events: List[NoteEvent] = []
    for row in reader:
        if not row:
            continue
        try:
            start_ms = int(float(row["time[ms]"]))
            duration_ms = int(float(row["length[ms]"]))
            pitch = str(row["pitch"]).strip()
            if duration_ms <= 0:
                continue
            events.append(NoteEvent(start_s=start_ms / 1000.0, duration_s=duration_ms / 1000.0, pitch=pitch))
        except Exception:
            continue
    return events


def pitch_to_midi(pitch: str) -> Optional[int]:
    try:
        s = pitch.strip()
        if not s:
            return None
        # Extract letter, accidental, octave
        letter = s[0].upper()
        idx = 1
        accidental = ''
        if idx < len(s) and s[idx] in ['#', 'b', 'B']:
            accidental = '#' if s[idx] == '#' else 'b'
            idx += 1
        octave = int(s[idx:])
        base_map: Dict[str, int] = {
            'C': 0,
            'D': 2,
            'E': 4,
            'F': 5,
            'G': 7,
            'A': 9,
            'B': 11,
        }
        semitone = base_map.get(letter)
        if semitone is None:
            return None
        if accidental == '#':
            semitone += 1
        elif accidental == 'b':
            semitone -= 1
        midi_num = (octave + 1) * 12 + (semitone % 12)
        return midi_num
    except Exception:
        return None


def group_events_no_overlap(events: List[NoteEvent]) -> List[Tuple[float, float, str]]:
    # Group by integer millisecond start to avoid float equality issues
    groups: Dict[int, List[NoteEvent]] = {}
    for ev in events:
        start_ms = int(round(ev.start_s * 1000))
        groups.setdefault(start_ms, []).append(ev)

    sorted_starts_ms = sorted(groups.keys())

    grouped: List[Tuple[float, float, str]] = []
    for i, start_ms in enumerate(sorted_starts_ms):
        group_events = groups[start_ms]
        # Build label: unique pitches, keyboard order (by MIDI), fallback to alpha
        unique_pitches = sorted(set(ev.pitch for ev in group_events))
        def sort_key(p: str) -> Tuple[int, str]:
            m = pitch_to_midi(p)
            return (m if m is not None else 10_000, p)
        ordered = sorted(unique_pitches, key=sort_key)
        label = ", ".join(ordered)

        start_s = start_ms / 1000.0
        # Intended end = max end among events in group
        intended_end_s = max(ev.start_s + ev.duration_s for ev in group_events)
        # Cut at next group's start if any
        if i + 1 < len(sorted_starts_ms):
            next_start_s = sorted_starts_ms[i + 1] / 1000.0
            end_s = min(intended_end_s, next_start_s)
        else:
            end_s = intended_end_s

        # Avoid non-positive durations
        if end_s > start_s:
            grouped.append((start_s, end_s - start_s, label))
    return grouped


def build_video(
    events: List[NoteEvent],
    size: Tuple[int, int],
    output_path: str,
    fps: int,
    bg_color: Tuple[int, int, int],
    font_name: Optional[str],
    font_size: Optional[int],
    fade_ms: int,
    tail_ms: int,
    text_color: Tuple[int, int, int],
    bgm_path: Optional[str],
    bgm_volume: float,
) -> None:
    width, height = size
    if font_size is None:
        font_size = max(12, int(height * 0.2))

    font = resolve_font(font_name, font_size)

    # Compute base duration from events
    base_total_s = 0.0
    for ev in events:
        base_total_s = max(base_total_s, ev.start_s + ev.duration_s)
    base_total_s += max(0, tail_ms) / 1000.0

    audio_clip: Optional[AudioFileClip] = None
    video_total_s = base_total_s

    if bgm_path:
        try:
            audio_clip = AudioFileClip(bgm_path)
            if bgm_volume != 1.0:
                audio_clip = audio_clip.volumex(bgm_volume)  # type: ignore[assignment]
            bgm_duration = float(getattr(audio_clip, "duration", 0.0))
            video_total_s = max(base_total_s, bgm_duration)
        except Exception as exc:
            raise SystemExit(f"Failed to load BGM '{bgm_path}': {exc}")

    background = ColorClip(size, color=bg_color).set_duration(video_total_s)

    # Generate non-overlapping grouped clips
    groups = group_events_no_overlap(events)

    text_clips: List[ImageClip] = []
    fade_s_default = max(0.0, fade_ms / 1000.0)

    for start_s, duration_s, label in groups:
        text_img = render_text_image(label, font, text_color)
        text_np = np.array(text_img)
        clip = ImageClip(text_np).set_position("center").set_start(start_s).set_duration(duration_s)
        fade_s = min(fade_s_default, duration_s / 2.0)
        if fade_s > 0:
            clip = clip.fx(vfx.fadein, fade_s).fx(vfx.fadeout, fade_s)
        text_clips.append(clip)

    composite = CompositeVideoClip([background, *text_clips], size=size)

    if audio_clip is not None:
        composite = composite.set_audio(audio_clip)

    try:
        composite.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio=audio_clip is not None,
            audio_codec="aac" if audio_clip is not None else None,
            audio_bitrate="192k" if audio_clip is not None else None,
            preset="medium",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            verbose=False,
            logger=None,
        )
    finally:
        try:
            composite.close()
        except Exception:
            pass
        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Render note events CSV (stdin) to a video with fading pitch text.")
    parser.add_argument("-o", "--output", required=True, help="Output video file path (e.g., out.mp4)")
    parser.add_argument("--size", default="480x270", type=parse_size, help="Video size WxH (default: 480x270)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--bg", type=parse_color, default="#000000", help="Background color (default: #000000)")
    parser.add_argument("--font", default=None, help="Font name or path (default: system default)")
    parser.add_argument("--font-size", type=int, default=None, help="Font size in px (default: ~20% of height)")
    parser.add_argument("--fade-ms", type=int, default=120, help="Fade-in/out duration in ms (default: 120)")
    parser.add_argument("--tail-ms", type=int, default=500, help="Tail margin after last event in ms (default: 500)")
    parser.add_argument("--text-color", type=parse_color, default="#FFFFFF", help="Text color (default: #FFFFFF)")
    parser.add_argument("--bgm", default=None, help="Optional background audio path (e.g., .mp3)")
    parser.add_argument("--bgm-volume", type=float, default=1.0, help="BGM volume multiplier (default: 1.0)")

    args = parser.parse_args()

    events = read_events_from_stdin()
    if not events:
        raise SystemExit("No events parsed from stdin.")

    build_video(
        events=events,
        size=args.size,
        output_path=args.output,
        fps=args.fps,
        bg_color=args.bg,
        font_name=args.font,
        font_size=args.font_size,
        fade_ms=args.fade_ms,
        tail_ms=args.tail_ms,
        text_color=args.text_color,
        bgm_path=args.bgm,
        bgm_volume=args.bgm_volume,
    )


# Usage: cat notes.csv | python csv_to_video.py -o out.mp4 --size 480x270 --bgm song.mp3
if __name__ == "__main__":
    main() 