#!/usr/bin/env python3
"""
Muse EEG-only WebSocket bridge for eeg_driving_game_v7_motor_imagery.html.

Streams raw EEG samples only. It deliberately ignores accelerometer and gyro streams.

Install if needed:
    pip install pylsl websockets

Run:
    python muse_bridge_eeg_only_v7.py

Then open the v7 HTML file, choose "Muse EEG bridge", click Connect, and calibrate rest/left/right.
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Tuple

import websockets
from pylsl import StreamInlet, resolve_byprop

DEFAULT_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def normalise_label(label: str) -> str:
    label = (label or "").strip().upper()
    aliases = {
        "TP9": "TP9",
        "TP10": "TP10",
        "AF7": "AF7",
        "AF8": "AF8",
        "EEG TP9": "TP9",
        "EEG TP10": "TP10",
        "EEG AF7": "AF7",
        "EEG AF8": "AF8",
    }
    return aliases.get(label, label.replace("EEG_", "").replace("EEG ", ""))


def get_channel_labels(inlet: StreamInlet) -> List[str]:
    info = inlet.info()
    labels: List[str] = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(normalise_label(ch.child_value("label")))
            ch = ch.next_sibling()
    except Exception:
        labels = []
    if not labels or all(not x for x in labels):
        labels = DEFAULT_CHANNELS[: info.channel_count()]
    return labels


def make_index(labels: List[str]) -> Dict[str, int]:
    labels_norm = [normalise_label(x) for x in labels]
    idx = {}
    for target in DEFAULT_CHANNELS:
        if target in labels_norm:
            idx[target] = labels_norm.index(target)
    if len(idx) < 4 and len(labels) >= 4:
        # Muse often arrives in TP9, AF7, AF8, TP10 order.
        idx = {name: i for i, name in enumerate(DEFAULT_CHANNELS)}
    missing = [c for c in DEFAULT_CHANNELS if c not in idx]
    if missing:
        raise RuntimeError(f"Could not find channels {missing}. Stream labels were: {labels}")
    return idx


def find_eeg_stream(timeout: float = 12.0):
    print("Searching for LSL EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        raise RuntimeError("No LSL EEG stream found. Start MuseLSL/BlueMuse first.")
    # Prefer a Muse stream, but fall back to the first EEG stream.
    for s in streams:
        name = (s.name() or "").lower()
        if "muse" in name:
            return s
    return streams[0]


async def stream_eeg(websocket, path=None, *, inlet: StreamInlet, idx: Dict[str, int], fs: float, chunk_size: int):
    print("Client connected")
    try:
        while True:
            samples, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=chunk_size)
            if not samples:
                await asyncio.sleep(0.005)
                continue
            rows: List[List[float]] = []
            for row in samples:
                rows.append([
                    float(row[idx["TP9"]]),
                    float(row[idx["AF7"]]),
                    float(row[idx["AF8"]]),
                    float(row[idx["TP10"]]),
                ])
            message = {
                "source": "eeg_raw",
                "fs": fs,
                "channels": DEFAULT_CHANNELS,
                "samples": rows,
                "t": time.time(),
            }
            await websocket.send(json.dumps(message))
    except websockets.ConnectionClosed:
        print("Client disconnected")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    stream = find_eeg_stream()
    inlet = StreamInlet(stream, max_buflen=8)
    labels = get_channel_labels(inlet)
    idx = make_index(labels)
    fs = float(stream.nominal_srate() or 256)

    print(f"Using EEG stream: {stream.name()} | fs={fs:g} Hz | labels={labels}")
    print(f"WebSocket EEG-only bridge running at ws://{args.host}:{args.port}")

    async def handler(websocket, path=None):
        await stream_eeg(websocket, path, inlet=inlet, idx=idx, fs=fs, chunk_size=args.chunk_size)

    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
