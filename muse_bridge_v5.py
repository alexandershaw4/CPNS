#!/usr/bin/env python3
"""
Muse -> WebSocket bridge for the Neuroadaptive Driving Game.

v5: adds longer configurable calibration, adaptive neutral correction, and safer steering decay.

Recommended running order:

Terminal 1:
    muselsl stream --acc

Terminal 2:
    python muse_bridge_v5.py --source lsl --control-mode motion --calibrate --debug-motion

Browser:
    Input source -> Muse WebSocket bridge -> Connect Muse Bridge

The bridge sends:
    {"focus": 0.72, "precision": 0.84, "steer": -0.15, "blink": false}

Control design:
- Muse ACC / head tilt        -> left-right steering
- EEG alpha/theta             -> focus / boost
- EEG beta/alpha + artefact   -> precision / handling stability
- blink / large frontal pulse -> brake

Dependencies:
    pip install websockets numpy pylsl scipy
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
from typing import AsyncGenerator, Optional

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
    def __init__(self, alpha: float = 0.10):
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
    width = max(width, 1e-9)
    return sigmoid((value - centre) / width)


@dataclass
class CalibrationProfile:
    neutral_asym: float = 0.0
    left_asym: float = -0.1
    right_asym: float = 0.1

    neutral_acc: float = 0.0
    left_acc: float = -0.25
    right_acc: float = 0.25
    has_acc: bool = False
    acc_axis: int = 0
    acc_deadzone: float = 0.08
    max_steer: float = 0.85

    blink_threshold: float = 500.0
    focus_low: float = 0.25
    focus_high: float = 0.75
    precision_low: float = 0.25
    precision_high: float = 0.75
    quality_floor: float = 0.20

    def steer_from_asymmetry(self, asymmetry: float, gain: float = 1.0) -> float:
        return self._piecewise(asymmetry, self.neutral_asym, self.left_asym, self.right_asym, gain, deadzone=0.02)

    def steer_from_acc(self, acc: float, gain: float = 1.0) -> float:
        return self._piecewise(
            acc,
            self.neutral_acc,
            self.left_acc,
            self.right_acc,
            gain,
            deadzone=self.acc_deadzone,
            max_abs=self.max_steer,
        )

    @staticmethod
    def _piecewise(
        value: float,
        neutral: float,
        left: float,
        right: float,
        gain: float,
        deadzone: float = 0.0,
        max_abs: float = 1.0,
    ) -> float:
        value = float(value)

        if abs(value - neutral) <= abs(deadzone):
            return 0.0

        dl = left - neutral
        dr = right - neutral

        # If calibration produced a tiny side displacement, avoid explosive gain.
        min_sep = 1e-3
        left_score = ((value - neutral) / dl) if abs(dl) > min_sep else 0.0
        right_score = ((value - neutral) / dr) if abs(dr) > min_sep else 0.0

        if left_score > 0 and left_score >= right_score:
            # subtract deadzone in normalised units so steering ramps from zero
            dz = abs(deadzone / dl) if abs(dl) > min_sep else 0.0
            return float(np.clip(-gain * max(0.0, left_score - dz), -max_abs, 0.0))
        if right_score > 0:
            dz = abs(deadzone / dr) if abs(dr) > min_sep else 0.0
            return float(np.clip(gain * max(0.0, right_score - dz), 0.0, max_abs))
        return 0.0

    def scale_focus(self, raw_focus: float) -> float:
        return float(np.clip((raw_focus - self.focus_low) / (self.focus_high - self.focus_low + 1e-9), 0.0, 1.0))

    def scale_precision(self, raw_precision: float) -> float:
        return float(np.clip((raw_precision - self.precision_low) / (self.precision_high - self.precision_low + 1e-9), 0.0, 1.0))


class MotionSteeringDecoder:
    """
    Head-tilt steering from Muse accelerometer.

    The important bit is the dead-zone. Without it, any small offset from neutral
    becomes a permanent steering command, which makes the car spin.
    """

    def __init__(
        self,
        calibration: Optional[CalibrationProfile] = None,
        axis: int = 0,
        gain: float = 1.4,
        smooth_alpha: float = 0.22,
        invert: bool = False,
        deadzone: float = 0.10,
        max_steer: float = 0.75,
    ):
        self.calibration = calibration
        self.axis = axis
        self.gain = gain
        self.smooth_alpha = smooth_alpha
        self.invert = invert
        self.deadzone = deadzone
        self.max_steer = max_steer
        self.neutral = 0.0
        self.output = 0.0
        self.initialised = False

    def update(self, acc_sample: Optional[np.ndarray]) -> float:
        axis = self.calibration.acc_axis if (self.calibration is not None and self.calibration.has_acc) else self.axis

        if acc_sample is None or len(acc_sample) <= axis:
            self.output = (1 - self.smooth_alpha) * self.output
            return float(np.clip(self.output, -self.max_steer, self.max_steer))

        value = float(acc_sample[axis])
        if self.invert:
            value = -value

        if self.calibration is not None and self.calibration.has_acc:
            # Use the learned neutral/left/right map, but let neutral drift very slowly
            # when the user is close to centre. This prevents a small posture shift
            # from becoming a permanent steering command after calibration.
            err_from_neutral = value - self.calibration.neutral_acc
            if abs(err_from_neutral) < max(self.calibration.acc_deadzone * 0.75, 1e-4):
                self.calibration.neutral_acc = 0.995 * self.calibration.neutral_acc + 0.005 * value
            target = self.calibration.steer_from_acc(value, gain=self.gain)
        else:
            if not self.initialised:
                self.neutral = value
                self.initialised = True
            err = value - self.neutral
            # Re-centre slowly when close to neutral.
            if abs(err) < self.deadzone:
                self.neutral = 0.98 * self.neutral + 0.02 * value
                target = 0.0
            else:
                target = np.sign(err) * max(0.0, abs(err) - self.deadzone) * self.gain
                target = np.clip(target, -self.max_steer, self.max_steer)

        alpha = self.smooth_alpha if abs(target) > 0 else max(self.smooth_alpha, 0.35)
        self.output = (1 - alpha) * self.output + alpha * float(target)
        if abs(self.output) < 0.03:
            self.output = 0.0
        return float(np.clip(self.output, -self.max_steer, self.max_steer))


class EEGSteeringDecoder:
    def __init__(
        self,
        calibration: Optional[CalibrationProfile] = None,
        gain: float = 2.0,
        baseline_alpha: float = 0.01,
        smooth_alpha: float = 0.18,
    ):
        self.calibration = calibration
        self.gain = gain
        self.baseline_alpha = baseline_alpha
        self.smooth_alpha = smooth_alpha
        self.mean = 0.0
        self.var = 0.02
        self.output = 0.0

    def update(self, asymmetry: float, quality: float = 1.0) -> float:
        asymmetry = float(np.clip(asymmetry, -1.0, 1.0))
        quality = float(np.clip(quality, 0.0, 1.0))

        if self.calibration is not None:
            target = self.calibration.steer_from_asymmetry(asymmetry, gain=self.gain)
            if quality < self.calibration.quality_floor:
                target *= 0.25
            self.output = (1 - self.smooth_alpha) * self.output + self.smooth_alpha * target
            return float(np.clip(self.output, -1.0, 1.0))

        if quality > 0.35 and abs(asymmetry - self.mean) < 0.45:
            err = asymmetry - self.mean
            self.mean += self.baseline_alpha * err
            self.var = (1 - self.baseline_alpha) * self.var + self.baseline_alpha * err * err

        z = (asymmetry - self.mean) / math.sqrt(self.var + 1e-6)
        target = float(np.clip(z / 2.0 * self.gain, -1.0, 1.0))
        self.output = (1 - self.smooth_alpha) * self.output + self.smooth_alpha * target
        return float(np.clip(self.output, -1.0, 1.0))


def parse_numeric_packet(text: str) -> Optional[list[float]]:
    text = text.strip()
    if not text:
        return None

    try:
        msg = json.loads(text)
        if isinstance(msg, dict):
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
    if samples.ndim != 2 or samples.shape[0] < 8:
        return ControlState(source=source)

    x = samples[-min(samples.shape[0], 256):]
    x = x - np.nanmedian(x, axis=0, keepdims=True)
    amp = np.nanmedian(np.abs(x), axis=0)
    global_amp = float(np.nanmedian(amp))
    quality = 1.0 - robust_scale(global_amp, centre=180.0, width=80.0)
    quality = float(np.clip(quality, 0.1, 1.0))

    recent = x[-8:]
    blink_score = float(np.nanmax(np.abs(recent[:, : min(2, x.shape[1])]))) if x.shape[1] else 0.0
    blink = blink_score > 500.0

    if x.shape[1] >= 4:
        left = float(np.nanmedian(np.abs(x[:, [0, 1]])))
        right = float(np.nanmedian(np.abs(x[:, [2, 3]])))
        steer = np.clip((right - left) / (right + left + 1e-9) * 2.5, -1, 1)
    else:
        steer = 0.0

    stability = 1.0 - robust_scale(global_amp, centre=120.0, width=60.0)
    focus = 0.35 + 0.55 * stability
    precision = 0.35 + 0.60 * quality

    return ControlState(focus, precision, float(steer), blink, quality, source).clipped()


async def mock_source(hz: float = 20.0) -> AsyncGenerator[ControlState, None]:
    print("Using MOCK source.")
    t0 = time.time()
    while True:
        t = time.time() - t0
        focus = 0.55 + 0.28 * math.sin(t * 0.75) + 0.06 * math.sin(t * 2.2)
        precision = 0.70 + 0.20 * math.sin(t * 0.37 + 1.1)
        steer = 0.45 * math.sin(t * 0.85)
        blink = random.random() < 0.006
        yield ControlState(focus, precision, steer, blink, 1.0, "mock").clipped()
        await asyncio.sleep(1.0 / hz)


async def udp_source(port: int = 5000, hz: float = 20.0) -> AsyncGenerator[ControlState, None]:
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


async def lsl_source(
    hz: float = 20.0,
    window_seconds: float = 2.0,
    calibrate: bool = False,
    control_mode: str = "motion",
    motion_axis: int = -1,
    motion_gain: float = 1.4,
    steer_gain: float = 2.0,
    invert_motion: bool = False,
    motion_deadzone: float = 0.10,
    max_steer: float = 0.75,
    debug_motion: bool = False,
    neutral_seconds: float = 10.0,
    side_seconds: float = 8.0,
    blink_seconds: float = 6.0,
) -> AsyncGenerator[ControlState, None]:
    try:
        from pylsl import StreamInlet, resolve_byprop
        from scipy.signal import welch
    except ImportError as exc:
        raise RuntimeError("LSL mode requires: pip install pylsl scipy") from exc

    print("Searching for LSL EEG stream...")
    eeg_streams = resolve_byprop("type", "EEG", timeout=10)
    if not eeg_streams:
        raise RuntimeError("No LSL EEG stream found. Start muselsl first.")

    eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=10)
    eeg_info = eeg_inlet.info()
    fs = float(eeg_info.nominal_srate()) if eeg_info.nominal_srate() > 0 else 256.0
    n_channels = int(eeg_info.channel_count())
    print(f"Connected to LSL EEG stream: {eeg_info.name()} | fs={fs:g} Hz | channels={n_channels}")

    print("Searching for LSL ACC stream...")
    acc_streams = resolve_byprop("type", "ACC", timeout=3)
    acc_inlet = None
    acc_channels = 0
    latest_acc: Optional[np.ndarray] = None
    if acc_streams:
        acc_inlet = StreamInlet(acc_streams[0], max_buflen=10)
        acc_info = acc_inlet.info()
        acc_channels = int(acc_info.channel_count())
        print(f"Connected to LSL ACC stream: {acc_info.name()} | channels={acc_channels}")
    else:
        print("No ACC stream found. Start Muse with: muselsl stream --acc")
        print("Falling back to EEG steering only.")

    n_window = int(window_seconds * fs)
    eeg_buffer = deque(maxlen=max(n_window, 64))
    acc_buffer = deque(maxlen=512)

    smoother = FeatureSmoother(alpha=0.15)
    calibration: Optional[CalibrationProfile] = None
    eeg_decoder = EEGSteeringDecoder(gain=steer_gain, smooth_alpha=0.22)
    motion_decoder = MotionSteeringDecoder(axis=max(0, motion_axis), gain=motion_gain, smooth_alpha=0.22, invert=invert_motion, deadzone=motion_deadzone, max_steer=max_steer)

    def bandpower(x: np.ndarray, lo: float, hi: float) -> float:
        freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs)))
        idx = (freqs >= lo) & (freqs <= hi)
        return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0

    def drain_acc() -> Optional[np.ndarray]:
        nonlocal latest_acc
        if acc_inlet is None:
            return latest_acc
        while True:
            sample, _ts = acc_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            latest_acc = np.asarray(sample[:acc_channels], dtype=float)
            acc_buffer.append(latest_acc)
        return latest_acc

    def compute_features_from_buffer():
        if len(eeg_buffer) < max(64, int(0.75 * fs)):
            return None

        arr = np.asarray(eeg_buffer, dtype=float)
        arr = arr - np.nanmedian(arr, axis=0, keepdims=True)

        theta = np.median([bandpower(arr[:, ch], 4, 7) for ch in range(arr.shape[1])])
        alpha = np.median([bandpower(arr[:, ch], 8, 12) for ch in range(arr.shape[1])])
        beta = np.median([bandpower(arr[:, ch], 13, 30) for ch in range(arr.shape[1])])

        alpha_theta = alpha / (theta + 1e-9)
        beta_alpha = beta / (alpha + 1e-9)

        focus = robust_scale(alpha_theta, centre=1.2, width=0.55)
        precision = 1.0 - robust_scale(beta_alpha, centre=1.8, width=0.7)

        amp = float(np.nanmax(np.abs(arr[-int(0.20 * fs):, : min(2, arr.shape[1])]))) if arr.shape[1] else 0.0
        quality = 1.0 - robust_scale(float(np.nanmedian(np.abs(arr))), centre=150.0, width=80.0)

        if arr.shape[1] >= 4:
            af7_alpha = bandpower(arr[:, 1], 8, 12)
            af8_alpha = bandpower(arr[:, 2], 8, 12)
            tp9_alpha = bandpower(arr[:, 0], 8, 12)
            tp10_alpha = bandpower(arr[:, 3], 8, 12)
            frontal_asym = (af8_alpha - af7_alpha) / (af8_alpha + af7_alpha + 1e-9)
            temporal_asym = (tp10_alpha - tp9_alpha) / (tp10_alpha + tp9_alpha + 1e-9)
            raw_asym = 0.80 * frontal_asym + 0.20 * temporal_asym
        else:
            raw_asym = 0.0

        return {
            "focus": float(focus),
            "precision": float(precision),
            "amp": float(amp),
            "quality": float(quality),
            "raw_asym": float(raw_asym),
        }

    async def warmup(seconds: float = 1.5):
        print(f"Warming up buffers for {seconds:g} seconds...")
        t_end = time.time() + seconds
        while time.time() < t_end:
            sample, _ts = eeg_inlet.pull_sample(timeout=0.01)
            if sample is not None:
                eeg_buffer.append([float(v) for v in sample[:n_channels]])
            drain_acc()
            await asyncio.sleep(0.001)

    async def collect_calibration_block(label: str, seconds: float = 7.0):
        print()
        print(f"Calibration: {label} for {seconds:g} seconds...")
        t_end = time.time() + seconds
        asym_values, focus_values, precision_values, amp_values, quality_values, acc_values = [], [], [], [], [], []

        while time.time() < t_end:
            sample, _ts = eeg_inlet.pull_sample(timeout=0.01)
            if sample is not None:
                eeg_buffer.append([float(v) for v in sample[:n_channels]])

            acc = drain_acc()
            if acc is not None and len(acc) >= 3:
                acc_vec = np.asarray(acc[:3], dtype=float)
                if invert_motion:
                    acc_vec = -acc_vec
                acc_values.append(acc_vec)

            feats = compute_features_from_buffer()
            if feats is not None:
                asym_values.append(feats["raw_asym"])
                focus_values.append(feats["focus"])
                precision_values.append(feats["precision"])
                amp_values.append(feats["amp"])
                quality_values.append(feats["quality"])

            await asyncio.sleep(0.02)

        def med(xs, fallback=0.0):
            return float(np.nanmedian(xs)) if len(xs) else fallback

        return {
            "asym": med(asym_values, 0.0),
            "focus": med(focus_values, 0.5),
            "precision": med(precision_values, 0.7),
            "amp95": float(np.nanpercentile(amp_values, 95)) if len(amp_values) else 500.0,
            "quality": med(quality_values, 0.8),
            "acc": np.nanmedian(np.vstack(acc_values), axis=0) if len(acc_values) else np.array([0.0, 0.0, 0.0]),
            "has_acc": len(acc_values) > 10,
        }

    if calibrate:
        print()
        print("=== Muse driving calibration ===")
        print("This learns neutral head position, left tilt, right tilt, blink threshold, and EEG ranges.")
        await warmup(1.5)

        input("Press Enter, then hold NEUTRAL: eyes forward, head straight...")
        neutral = await collect_calibration_block("NEUTRAL: head straight, relaxed face", seconds=neutral_seconds)

        input("Press Enter, then hold LEFT: gently tilt/lean head left, no big movement...")
        left = await collect_calibration_block("LEFT: head tilt left", seconds=side_seconds)

        input("Press Enter, then hold RIGHT: gently tilt/lean head right, no big movement...")
        right = await collect_calibration_block("RIGHT: head tilt right", seconds=side_seconds)

        input("Press Enter, then BLINK several times...")
        blink_block = await collect_calibration_block("BLINK: deliberate blinks", seconds=blink_seconds)

        focus_vals = sorted([neutral["focus"], left["focus"], right["focus"]])
        precision_vals = sorted([neutral["precision"], left["precision"], right["precision"]])
        blink_threshold = max(500.0, neutral["amp95"] * 1.75, blink_block["amp95"] * 0.55)

        has_acc = bool(neutral["has_acc"] and left["has_acc"] and right["has_acc"])
        if has_acc:
            # Auto-pick the axis whose left/right calibration produced the largest
            # usable separation. This avoids guessing whether Muse roll is x/y/z.
            if motion_axis in (0, 1, 2):
                chosen_axis = motion_axis
            else:
                sep = np.abs(np.asarray(left["acc"]) - np.asarray(right["acc"]))
                chosen_axis = int(np.nanargmax(sep))

            neutral_acc = float(neutral["acc"][chosen_axis])
            left_acc = float(left["acc"][chosen_axis])
            right_acc = float(right["acc"][chosen_axis])
            side_sep = max(abs(left_acc - neutral_acc), abs(right_acc - neutral_acc), 1e-3)
            acc_deadzone = max(motion_deadzone, 0.22 * side_sep)
        else:
            chosen_axis = max(0, motion_axis)
            neutral_acc, left_acc, right_acc = 0.0, -0.25, 0.25
            acc_deadzone = motion_deadzone

        calibration = CalibrationProfile(
            neutral_asym=neutral["asym"],
            left_asym=left["asym"],
            right_asym=right["asym"],
            neutral_acc=neutral_acc,
            left_acc=left_acc,
            right_acc=right_acc,
            has_acc=has_acc,
            acc_axis=chosen_axis,
            acc_deadzone=acc_deadzone,
            max_steer=max_steer,
            blink_threshold=blink_threshold,
            focus_low=max(0.0, focus_vals[0] - 0.05),
            focus_high=min(1.0, focus_vals[-1] + 0.05),
            precision_low=max(0.0, precision_vals[0] - 0.05),
            precision_high=min(1.0, precision_vals[-1] + 0.05),
            quality_floor=max(0.10, min(0.45, neutral["quality"] * 0.55)),
        )

        eeg_decoder = EEGSteeringDecoder(calibration=calibration, gain=steer_gain, smooth_alpha=0.24)
        motion_decoder = MotionSteeringDecoder(
            calibration=calibration,
            axis=calibration.acc_axis,
            gain=motion_gain,
            smooth_alpha=0.22,
            invert=invert_motion,
            deadzone=motion_deadzone,
            max_steer=max_steer,
        )

        print()
        print("Calibration complete:")
        print(f"  ACC available={calibration.has_acc} | chosen axis={calibration.acc_axis} | invert={invert_motion}")
        print(f"  acc neutral={calibration.neutral_acc:+0.4f} left={calibration.left_acc:+0.4f} right={calibration.right_acc:+0.4f} deadzone={calibration.acc_deadzone:0.4f}")
        print(f"  EEG asym neutral={calibration.neutral_asym:+0.4f} left={calibration.left_asym:+0.4f} right={calibration.right_asym:+0.4f}")
        print(f"  blink threshold={calibration.blink_threshold:0.1f}")
        print(f"  focus range={calibration.focus_low:0.2f}..{calibration.focus_high:0.2f}")
        print(f"  precision range={calibration.precision_low:0.2f}..{calibration.precision_high:0.2f}")
        print()
        if not calibration.has_acc:
            print("WARNING: no usable ACC calibration data. Start muselsl with --acc.")
        print("Start driving.")
        print()

    last_emit = 0.0

    while True:
        sample, _ts = eeg_inlet.pull_sample(timeout=0.0)
        if sample is not None:
            eeg_buffer.append([float(v) for v in sample[:n_channels]])

        acc = drain_acc()
        now = time.time()

        if now - last_emit >= 1.0 / hz:
            feats = compute_features_from_buffer()
            if feats is not None:
                focus = feats["focus"]
                precision = feats["precision"]
                if calibration is not None:
                    focus = calibration.scale_focus(focus)
                    precision = calibration.scale_precision(precision)

                quality = feats["quality"]
                blink = feats["amp"] > (calibration.blink_threshold if calibration is not None else 500.0)

                eeg_steer = eeg_decoder.update(feats["raw_asym"], quality=quality)
                motion_steer = motion_decoder.update(acc)

                if control_mode == "motion":
                    steer = motion_steer
                    source_name = "lsl+acc"
                elif control_mode == "eeg":
                    steer = eeg_steer
                    source_name = "lsl+eeg"
                elif control_mode == "hybrid":
                    steer = 0.85 * motion_steer + 0.15 * eeg_steer
                    source_name = "lsl+hyb"
                else:
                    steer = motion_steer
                    source_name = "lsl+acc"

                if debug_motion and acc is not None:
                    axis = calibration.acc_axis if (calibration is not None and calibration.has_acc) else max(0, motion_axis)
                    acc_vals = ",".join(f"{float(v):+0.3f}" for v in np.asarray(acc[:3], dtype=float))
                    print(f"\nACC=[{acc_vals}] axis={axis} motion_steer={motion_steer:+0.3f} eeg_steer={eeg_steer:+0.3f}")

                target = ControlState(focus, precision, float(steer), bool(blink), quality, source_name).clipped()
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
    p = argparse.ArgumentParser(description="Muse/LSL to WebSocket bridge for the EEG driving game")
    p.add_argument("--source", choices=["mock", "udp", "lsl"], default="mock")
    p.add_argument("--host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--udp-port", type=int, default=5000)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--calibrate", action="store_true", help="Run neutral/left/right/blink calibration before streaming")
    p.add_argument("--control-mode", choices=["motion", "eeg", "hybrid"], default="motion")
    p.add_argument("--motion-axis", type=int, default=-1, help="ACC axis for steering: -1 auto, or try 0, 1, 2")
    p.add_argument("--motion-gain", type=float, default=1.4, help="Head-tilt steering gain")
    p.add_argument("--steer-gain", type=float, default=2.0, help="EEG steering gain, used in eeg/hybrid mode")
    p.add_argument("--invert-motion", action="store_true", help="Invert motion steering direction")
    p.add_argument("--motion-deadzone", type=float, default=0.10, help="Neutral dead-zone for head-tilt steering")
    p.add_argument("--max-steer", type=float, default=0.75, help="Maximum absolute steering sent to the game")
    p.add_argument("--debug-motion", action="store_true", help="Print raw ACC values and steering diagnostics")
    p.add_argument("--neutral-seconds", type=float, default=10.0, help="Neutral calibration duration")
    p.add_argument("--side-seconds", type=float, default=8.0, help="Left/right calibration duration")
    p.add_argument("--blink-seconds", type=float, default=6.0, help="Blink calibration duration")
    return p


async def main_async(args: argparse.Namespace):
    if args.source == "mock":
        source = mock_source(hz=args.hz)
    elif args.source == "udp":
        source = udp_source(port=args.udp_port, hz=args.hz)
    elif args.source == "lsl":
        source = lsl_source(
            hz=args.hz,
            calibrate=args.calibrate,
            control_mode=args.control_mode,
            motion_axis=args.motion_axis,
            motion_gain=args.motion_gain,
            steer_gain=args.steer_gain,
            invert_motion=args.invert_motion,
            motion_deadzone=args.motion_deadzone,
            max_steer=args.max_steer,
            debug_motion=args.debug_motion,
            neutral_seconds=args.neutral_seconds,
            side_seconds=args.side_seconds,
            blink_seconds=args.blink_seconds,
        )
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
