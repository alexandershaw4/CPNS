#!/usr/bin/env python3
"""
Muse -> WebSocket bridge for the Neuroadaptive Driving Game.

This script sends real-time JSON messages to the browser game at:

    ws://localhost:8765

The browser expects messages like:

    {"focus": 0.72, "precision": 0.84, "steer": -0.15, "blink": false}

Start with mock mode:

    python muse_bridge.py --source mock

Then open the HTML game in Chrome, choose "Muse WebSocket bridge", and click
"Connect Muse Bridge".

Later options:

    python muse_bridge.py --source udp --udp-port 5000
    python muse_bridge.py --source lsl

Dependencies:

    pip install websockets numpy

Optional for LSL mode:

    pip install pylsl scipy

Notes:
- Browser pages cannot directly read UDP or LSL streams, so this local bridge
  translates Muse/UVic/LSL data into WebSocket JSON for the game.
- UDP mode is intentionally forgiving. If uvicMuse sends raw numeric packets,
  the bridge will try to parse the first four numeric channels as EEG.
- LSL mode looks for an EEG stream and computes a simple alpha/theta/beta
  feature set from a rolling buffer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import socket
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, Iterable, Optional

import numpy as np
import websockets


@dataclass
class ControlState:
    focus: float = 0.5
    precision: float = 0.7
    steer: float = 0.0
    blink: bool = False
    quality: float = 1.0
    source: str = "mock"
    timestamp: float = 0.0

    def clipped(self) -> "ControlState":
        self.focus = float(np.clip(self.focus, 0.0, 1.0))
        self.precision = float(np.clip(self.precision, 0.0, 1.0))
        self.steer = float(np.clip(self.steer, -1.0, 1.0))
        self.quality = float(np.clip(self.quality, 0.0, 1.0))
        self.timestamp = time.time()
        return self

    def to_json(self) -> str:
        self.clipped()
        return json.dumps(asdict(self))


class FeatureSmoother:
    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha
        self.state = ControlState()

    def update(self, target: ControlState) -> ControlState:
        a = self.alpha
        self.state.focus = (1 - a) * self.state.focus + a * target.focus
        self.state.precision = (1 - a) * self.state.precision + a * target.precision
        self.state.steer = (1 - a) * self.state.steer + a * target.steer
        self.state.quality = (1 - a) * self.state.quality + a * target.quality
        self.state.blink = bool(target.blink)
        self.state.source = target.source
        self.state.timestamp = time.time()
        return self.state.clipped()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def robust_scale(value: float, centre: float, width: float) -> float:
    """Map feature to 0..1 without assuming perfect calibration."""
    width = max(width, 1e-9)
    return sigmoid((value - centre) / width)


class EEGSteeringDecoder:
    """
    Convert noisy left-right EEG asymmetry into a usable controller.

    This uses an adaptive baseline, a dead-zone, hysteresis and smoothing.
    The output is not 'thought decoding' in the strong sense. It is a practical
    BCI-style left/neutral/right control signal based on sustained asymmetry.
    """

    def __init__(self, baseline_alpha: float = 0.01, smooth_alpha: float = 0.18):
        self.baseline_alpha = baseline_alpha
        self.smooth_alpha = smooth_alpha
        self.mean = 0.0
        self.var = 0.02
        self.output = 0.0
        self.state = 0

    def update(self, asymmetry: float, quality: float = 1.0) -> float:
        asymmetry = float(np.clip(asymmetry, -1.0, 1.0))
        quality = float(np.clip(quality, 0.0, 1.0))

        # Update baseline only when signal quality is reasonable and the current
        # asymmetry is not too extreme. This lets the neutral point drift slowly
        # with headset contact without absorbing deliberate left/right states.
        if quality > 0.35 and abs(asymmetry - self.mean) < 0.45:
            err = asymmetry - self.mean
            self.mean += self.baseline_alpha * err
            self.var = (1 - self.baseline_alpha) * self.var + self.baseline_alpha * err * err

        z = (asymmetry - self.mean) / math.sqrt(self.var + 1e-6)

        # Hysteretic left/right/neutral states. These thresholds are deliberately
        # conservative so the car does not jitter around the neutral point.
        enter = 1.15
        exit_ = 0.45
        if self.state == 0:
            if z > enter:
                self.state = 1
            elif z < -enter:
                self.state = -1
        elif self.state == 1 and z < exit_:
            self.state = 0
        elif self.state == -1 and z > -exit_:
            self.state = 0

        # Blend continuous z-score and discrete state. This gives a controller
        # that has both graded steering and clear left/right commitment.
        continuous = float(np.clip(z / 2.5, -1.0, 1.0))
        target = 0.65 * self.state + 0.35 * continuous
        self.output = smoothTowards(self.output, target, self.smooth_alpha)
        return float(np.clip(self.output, -1.0, 1.0))


def parse_numeric_packet(text: str) -> Optional[list[float]]:
    """Parse JSON arrays/objects or CSV-ish numeric strings."""
    text = text.strip()
    if not text:
        return None

    try:
        msg = json.loads(text)
        if isinstance(msg, dict):
            # If already in game-control format, handle elsewhere.
            if any(k in msg for k in ("focus", "precision", "steer", "blink")):
                return None
            values = []
            for key in ("TP9", "AF7", "AF8", "TP10", "ch1", "ch2", "ch3", "ch4"):
                if key in msg:
                    values.append(float(msg[key]))
            return values if values else None
        if isinstance(msg, list):
            return [float(v) for v in msg if isinstance(v, (int, float))]
    except Exception:
        pass

    cleaned = text.replace(";", ",").replace("\t", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            continue
    return nums if nums else None


def controls_from_raw_eeg(samples: np.ndarray, source: str = "udp") -> ControlState:
    """
    Very lightweight fallback feature mapping from recent raw EEG samples.

    samples shape: [time, channels]
    This is intentionally simple. The proper version should use calibrated
    bandpower over a rolling window once we know the exact stream format.
    """
    if samples.ndim != 2 or samples.shape[0] < 8:
        return ControlState(source=source)

    x = samples[-min(samples.shape[0], 256):]
    x = x - np.nanmedian(x, axis=0, keepdims=True)
    amp = np.nanmedian(np.abs(x), axis=0)
    global_amp = float(np.nanmedian(amp))

    # Crude quality: lower if enormous amplitudes suggest bad contact/artifact.
    quality = 1.0 - robust_scale(global_amp, centre=180.0, width=80.0)
    quality = float(np.clip(quality, 0.1, 1.0))

    # Crude blink detector: large frontal-ish transient if channels 1/2 are large.
    recent = x[-8:]
    blink_score = float(np.nanmax(np.abs(recent[:, : min(2, x.shape[1])]))) if x.shape[1] else 0.0
    blink = blink_score > 250.0

    # Crude left/right asymmetry if at least four channels exist.
    if x.shape[1] >= 4:
        left = float(np.nanmedian(np.abs(x[:, [0, 1]])))
        right = float(np.nanmedian(np.abs(x[:, [2, 3]])))
        steer = np.clip((right - left) / (right + left + 1e-9) * 2.5, -1, 1)
    else:
        steer = 0.0

    # Without proper spectral features, use stability as a playable proxy.
    stability = 1.0 - robust_scale(global_amp, centre=120.0, width=60.0)
    focus = 0.35 + 0.55 * stability
    precision = 0.35 + 0.60 * quality

    return ControlState(
        focus=focus,
        precision=precision,
        steer=float(steer),
        blink=blink,
        quality=quality,
        source=source,
    ).clipped()


async def mock_source(hz: float = 20.0) -> AsyncGenerator[ControlState, None]:
    print("Using MOCK source. This lets you test the browser bridge before connecting Muse.")
    t0 = time.time()
    while True:
        t = time.time() - t0
        focus = 0.55 + 0.28 * math.sin(t * 0.75) + 0.06 * math.sin(t * 2.2)
        precision = 0.70 + 0.20 * math.sin(t * 0.37 + 1.1)
        steer = 0.35 * math.sin(t * 0.85)
        blink = random.random() < 0.006
        yield ControlState(focus, precision, steer, blink, 1.0, "mock").clipped()
        await asyncio.sleep(1.0 / hz)


async def udp_source(port: int = 5000, hz: float = 20.0) -> AsyncGenerator[ControlState, None]:
    """
    Receive Muse/uvicMuse-style UDP packets and emit game controls.

    This supports either:
    1. Already-derived JSON controls, e.g.
       {"focus":0.7,"precision":0.8,"steer":-0.2,"blink":false}
    2. Numeric raw-ish EEG packets, CSV or JSON array.
    """
    print(f"Using UDP source on 127.0.0.1:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.setblocking(False)

    buffer: deque[list[float]] = deque(maxlen=512)
    loop = asyncio.get_running_loop()
    last_emit = 0.0

    while True:
        try:
            data, _addr = await loop.sock_recvfrom(sock, 8192)
            text = data.decode("utf-8", errors="ignore").strip()

            # If some upstream tool already sends game-control JSON, pass it through.
            try:
                msg = json.loads(text)
                if isinstance(msg, dict) and any(k in msg for k in ("focus", "precision", "steer", "blink")):
                    yield ControlState(
                        focus=float(msg.get("focus", 0.5)),
                        precision=float(msg.get("precision", 0.7)),
                        steer=float(msg.get("steer", 0.0)),
                        blink=bool(msg.get("blink", False)),
                        quality=float(msg.get("quality", 1.0)),
                        source="udp-json",
                    ).clipped()
                    continue
            except Exception:
                pass

            nums = parse_numeric_packet(text)
            if nums and len(nums) >= 4:
                buffer.append(nums[:4])

            now = time.time()
            if now - last_emit >= 1.0 / hz and len(buffer) >= 16:
                arr = np.asarray(buffer, dtype=float)
                yield controls_from_raw_eeg(arr, source="udp-raw")
                last_emit = now

        except BlockingIOError:
            await asyncio.sleep(0.001)
        except Exception as exc:
            print(f"UDP parse error: {exc}")
            await asyncio.sleep(0.01)


async def lsl_source(hz: float = 20.0, window_seconds: float = 2.0) -> AsyncGenerator[ControlState, None]:
    """
    Optional LSL source. Looks for an EEG stream and computes bandpower features.
    Requires pylsl and scipy.
    """
    try:
        from pylsl import StreamInlet, resolve_byprop
        from scipy.signal import welch
    except ImportError as exc:
        raise RuntimeError("LSL mode requires: pip install pylsl scipy") from exc

    print("Searching for LSL EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=10)
    if not streams:
        raise RuntimeError("No LSL EEG stream found. Start uvicMuse or muselsl first.")

    inlet = StreamInlet(streams[0], max_buflen=10)
    info = inlet.info()
    fs = float(info.nominal_srate()) if info.nominal_srate() > 0 else 256.0
    n_channels = int(info.channel_count())
    print(f"Connected to LSL EEG stream: {info.name()} | fs={fs:g} Hz | channels={n_channels}")

    n_window = int(window_seconds * fs)
    buffer = deque(maxlen=max(n_window, 64))
    smoother = FeatureSmoother(alpha=0.15)
    steering_decoder = EEGSteeringDecoder(baseline_alpha=0.01, smooth_alpha=0.22)

    def bandpower(x: np.ndarray, lo: float, hi: float) -> float:
        freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs)))
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

    last_emit = 0.0

    while True:
        sample, _ts = inlet.pull_sample(timeout=0.0)
        if sample is not None:
            buffer.append([float(v) for v in sample[:n_channels]])

        now = time.time()
        if now - last_emit >= 1.0 / hz and len(buffer) >= max(64, int(0.75 * fs)):
            arr = np.asarray(buffer, dtype=float)
            arr = arr - np.nanmedian(arr, axis=0, keepdims=True)

            # Use median across channels for robust global spectral indices.
            theta = np.median([bandpower(arr[:, ch], 4, 7) for ch in range(arr.shape[1])])
            alpha = np.median([bandpower(arr[:, ch], 8, 12) for ch in range(arr.shape[1])])
            beta = np.median([bandpower(arr[:, ch], 13, 30) for ch in range(arr.shape[1])])

            alpha_theta = alpha / (theta + 1e-9)
            beta_alpha = beta / (alpha + 1e-9)

            focus = robust_scale(alpha_theta, centre=1.2, width=0.55)
            precision = 1.0 - robust_scale(beta_alpha, centre=1.8, width=0.7)

            # Muse-ish channel convention often: TP9, AF7, AF8, TP10.
            if arr.shape[1] >= 4:
                # Muse channel order from muselsl is usually TP9, AF7, AF8, TP10, AUX.
                # For steering, use frontal alpha asymmetry first because AF7/AF8 are
                # more likely to pick up eye/attention/lateral artefact differences.
                # Fallback temporal channels are blended in lightly.
                af7_alpha = bandpower(arr[:, 1], 8, 12)
                af8_alpha = bandpower(arr[:, 2], 8, 12)
                tp9_alpha = bandpower(arr[:, 0], 8, 12)
                tp10_alpha = bandpower(arr[:, 3], 8, 12)

                frontal_asym = (af8_alpha - af7_alpha) / (af8_alpha + af7_alpha + 1e-9)
                temporal_asym = (tp10_alpha - tp9_alpha) / (tp10_alpha + tp9_alpha + 1e-9)
                raw_asym = 0.80 * frontal_asym + 0.20 * temporal_asym
                steer = steering_decoder.update(raw_asym, quality=quality)
            else:
                steer = 0.0

            amp = float(np.nanmax(np.abs(arr[-int(0.20 * fs):, : min(2, arr.shape[1])])) if arr.shape[1] else 0.0)
            blink = amp > 180.0
            quality = 1.0 - robust_scale(float(np.nanmedian(np.abs(arr))), centre=150.0, width=80.0)

            target = ControlState(focus, precision, float(steer), blink, quality, "lsl").clipped()
            yield smoother.update(target)
            last_emit = now

        await asyncio.sleep(0.001)


class BridgeServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: set = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        print(f"Browser connected. Clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"Browser disconnected. Clients: {len(self.clients)}")

    async def broadcast(self, message: str):
        if not self.clients:
            return
        stale = []
        for client in list(self.clients):
            try:
                await client.send(message)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def run(self, source: AsyncGenerator[ControlState, None]):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket bridge running at ws://{self.host}:{self.port}")
            print("Open the game, choose 'Muse WebSocket bridge', then click Connect.")
            async for state in source:
                await self.broadcast(state.to_json())
                print(
                    f"\r{state.source:8s} "
                    f"focus={state.focus:0.2f} "
                    f"precision={state.precision:0.2f} "
                    f"steer={state.steer:+0.2f} "
                    f"blink={int(state.blink)} "
                    f"quality={state.quality:0.2f}",
                    end="",
                    flush=True,
                )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Muse/UVic/LSL to WebSocket bridge for the EEG driving game")
    p.add_argument("--source", choices=["mock", "udp", "lsl"], default="mock")
    p.add_argument("--host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--udp-port", type=int, default=5000)
    p.add_argument("--hz", type=float, default=20.0)
    return p


async def main_async(args: argparse.Namespace):
    if args.source == "mock":
        source = mock_source(hz=args.hz)
    elif args.source == "udp":
        source = udp_source(port=args.udp_port, hz=args.hz)
    elif args.source == "lsl":
        source = lsl_source(hz=args.hz)
    else:
        raise ValueError(args.source)

    server = BridgeServer(args.host, args.ws_port)
    await server.run(source)


def main():
    args = build_argparser().parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
