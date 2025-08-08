# midi-pitch-visualizer

<img src="res/screenshot.png" alt="screenshot" width="360" />

A tiny toolkit to turn a melody in a MIDI file into a text-overlay practice video for the [Otamatone](https://otamatone.jp).

- `midi_to_csv.py`: Parse a single-track MIDI and print a CSV of note events to stdout.
- `csv_to_video.py`: Read the CSV from stdin and render an MP4 showing pitch names centered on screen at the right times.

## Requirements

- Python 3.9+
- ffmpeg installed and available on PATH
- Python libs: `mido`, `moviepy`, `Pillow`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python midi_to_csv.py path/to/input.mid | \
python csv_to_video.py -o out.mp4
```

## Common options

```bash
python midi_to_csv.py song.mid | \
python csv_to_video.py -o song.mp4 \
  --size 720x720 \
  --bg "#87aedd" \
  --font "Helvetica" \
  --font-size 120 \
  --fade-ms 120 \
  --tail-ms 800 \
  --text-color "#FFFFFF" \
  --bgm path/to/audio.mp3 \
  --bgm-volume 0.9
```

## `csv_to_video.py` options

- `--size WxH`: frame size (default 480x270)
- `--fps N`: frames per second (default 30)
- `--bg COLOR`: background color (e.g., `#000000`)
- `--font NAME|PATH`: font name or file
- `--font-size PX`: font size (default ≈20% of height)
- `--text-color COLOR`: text color
- `--fade-ms MS`: fade in/out ms per label
- `--tail-ms MS`: tail margin after the last event
- `--bgm PATH`: optional background audio
- `--bgm-volume RATIO`: BGM gain multiplier
- `--dynamic-font-size LOWER:UPPER`: per-note scaling from velocity (e.g., `0.6:1.6`)

## I/O formats

- MIDI input (assumptions):
  - Only track 0 is processed
  - Tempo: only `set_tempo` at t=0 is applied; later changes are ignored. If missing, 120 BPM is assumed
- Intermediate CSV (with header):

```csv
time[ms],pitch,length[ms],velocity
249,B5,1755,90
2157,B5,48,69
```

- Output: MP4 (H.264 yuv420p). If BGM is provided, audio is muxed as AAC

## Notes & limitations

- Simultaneous notes are shown side-by-side as comma-joined labels
- Sustain and some control changes are ignored
- Large inputs increase render time
- Rarely, fonts/ligatures may clip depending on the selected font

## How it works

1. MIDI → CSV: compute `time[ms]`, `length[ms]`, `pitch` (flat names preferred), `velocity`
2. CSV → Video: render centered text clips with fades; optionally add BGM
