#!/usr/bin/env python3
"""
MIDI→CSV extractor

Spec:
- Input: Single MIDI file (only track 0 is processed)
- Tempo: Fixed tempo only
  - Apply only set_tempo at t=0; ignore tempo changes afterwards
  - If no initial tempo: use 120 BPM (500000 µs/beat) and emit WARNING to stderr
- Notes:
  - Output one row per note (including simultaneous notes)
  - Pitch names: flats preferred (e.g., Db4, Gb3)
  - Velocity: use Note On velocity (0–127)
  - Sustain pedal (CC64): ignored
  - Unmatched Note On at EOF: closed at final time
- Timing:
  - time[ms]: start time from 0, rounded to nearest ms
  - length[ms]: duration, rounded to nearest ms
- Output: CSV to stdout with header: time[ms], pitch, length[ms], velocity
- Sorting: rows sorted by (time[ms], pitch)

Output Example:

time[ms],pitch,length[ms],velocity
249,B5,1755,90
2157,B5,48,69
2306,B5,107,71
2411,Db6,301,90
2688,B5,127,84
2966,Ab5,149,86

Usage:
  python midi_to_csv.py path/to/input.mid > output.csv
Dependencies: mido
"""
import argparse
import csv
import logging
import sys
from collections import defaultdict
from typing import List, Tuple, Dict

import mido

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def midi_note_to_name(note_number: int) -> str:
    octave = note_number // 12 - 1
    return f"{NOTE_NAMES_FLAT[note_number % 12]}{octave}"


def parse_single_track_to_rows(midi_path: str) -> List[Tuple[int, str, int, int]]:
    mid = mido.MidiFile(midi_path)
    track = mid.tracks[0]
    ticks_per_beat = mid.ticks_per_beat

    tempo_us_per_beat = 500000
    seconds_per_tick = tempo_us_per_beat / 1_000_000.0 / ticks_per_beat

    started = False
    saw_initial_tempo = False
    current_time_s = 0.0

    active_notes: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    rows: List[Tuple[int, str, int, int]] = []

    for msg in track:
        delta_ticks = msg.time
        if delta_ticks:
            current_time_s += delta_ticks * seconds_per_tick
            started = True

        if msg.type == 'set_tempo':
            if not started:
                tempo_us_per_beat = msg.tempo
                seconds_per_tick = tempo_us_per_beat / 1_000_000.0 / ticks_per_beat
                saw_initial_tempo = True
            continue

        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note].append((current_time_s, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if active_notes[msg.note]:
                start_s, velocity = active_notes[msg.note].pop()
                start_ms = int(round(start_s * 1000))
                length_ms = int(round((current_time_s - start_s) * 1000))
                pitch_name = midi_note_to_name(msg.note)
                rows.append((start_ms, pitch_name, length_ms, int(velocity)))

    if not saw_initial_tempo:
        logging.warning('No initial set_tempo at t=0; using default 120 BPM (500000 µs/beat).')

    last_time_s = current_time_s
    for note_number, stack in active_notes.items():
        while stack:
            start_s, velocity = stack.pop()
            start_ms = int(round(start_s * 1000))
            length_ms = int(round((last_time_s - start_s) * 1000))
            pitch_name = midi_note_to_name(note_number)
            rows.append((start_ms, pitch_name, length_ms, int(velocity)))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def write_csv(rows: List[Tuple[int, str, int, int]]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(['time[ms]', 'pitch', 'length[ms]', 'velocity'])
    for row in rows:
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_midi')
    args = parser.parse_args()

    rows = parse_single_track_to_rows(args.input_midi)
    write_csv(rows)


# python midi_to_csv.py path/to/input.mid > output.csv
if __name__ == '__main__':
    main() 
