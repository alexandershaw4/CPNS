const circleCanvas = document.getElementById('circleCanvas');
const waveCanvas = document.getElementById('waveCanvas');
const signalCanvas = document.getElementById('signalCanvas');
const spectrumCanvas = document.getElementById('spectrumCanvas');

const speedSlider = document.getElementById('speedSlider');
const speedReadout = document.getElementById('speedReadout');
const pauseButton = document.getElementById('pauseButton');

const freqASlider = document.getElementById('freqASlider');
const freqBSlider = document.getElementById('freqBSlider');
const noiseSlider = document.getElementById('noiseSlider');
const freqAReadout = document.getElementById('freqAReadout');
const freqBReadout = document.getElementById('freqBReadout');
const noiseReadout = document.getElementById('noiseReadout');

let theta = 0;
let paused = false;
let lastTime = null;
let trace = [];
let signalSeed = makeNoise(256);

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor((rect.height || rect.width * 0.65) * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function cssSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  return { w: rect.width, h: rect.height || rect.width * 0.65 };
}

function drawAxes(ctx, w, h, x0, y0) {
  ctx.strokeStyle = 'rgba(27,31,36,0.22)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(24, y0);
  ctx.lineTo(w - 18, y0);
  ctx.moveTo(x0, 18);
  ctx.lineTo(x0, h - 24);
  ctx.stroke();
}

function drawCircle() {
  const ctx = setupCanvas(circleCanvas);
  const { w, h } = cssSize(circleCanvas);
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) * 0.34;
  const x = cx + r * Math.cos(theta);
  const y = cy - r * Math.sin(theta);

  drawAxes(ctx, w, h, cx, cy);

  ctx.strokeStyle = 'rgba(30,91,98,0.28)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(140,84,51,0.95)';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(x, y);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(30,91,98,0.95)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(x, cy);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(103,77,145,0.95)';
  ctx.beginPath();
  ctx.moveTo(x, cy);
  ctx.lineTo(x, y);
  ctx.stroke();

  ctx.fillStyle = '#1b1f24';
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();

  label(ctx, 'cos θ / real', cx + r * 0.25, cy + 22, 'rgba(30,91,98,0.95)');
  label(ctx, 'sin θ / imaginary', x + 8, (cy + y) / 2, 'rgba(103,77,145,0.95)');
}

function drawWaves() {
  const ctx = setupCanvas(waveCanvas);
  const { w, h } = cssSize(waveCanvas);
  ctx.clearRect(0, 0, w, h);
  const mid1 = h * 0.34;
  const mid2 = h * 0.70;
  const amp = h * 0.18;

  drawWaveAxis(ctx, w, mid1, 'cos θ / real');
  drawWaveAxis(ctx, w, mid2, 'sin θ / imaginary');

  trace.push(theta);
  if (trace.length > 260) trace.shift();

  drawTrace(ctx, trace, w, mid1, amp, Math.cos, 'rgba(30,91,98,0.95)');
  drawTrace(ctx, trace, w, mid2, amp, Math.sin, 'rgba(103,77,145,0.95)');
}

function drawWaveAxis(ctx, w, y, text) {
  ctx.strokeStyle = 'rgba(27,31,36,0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(30, y);
  ctx.lineTo(w - 20, y);
  ctx.stroke();
  label(ctx, text, 34, y - 12, 'rgba(27,31,36,0.72)');
}

function drawTrace(ctx, values, w, mid, amp, fn, colour) {
  ctx.strokeStyle = colour;
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((t, i) => {
    const x = 30 + i * ((w - 60) / 260);
    const y = mid - amp * fn(t);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function label(ctx, text, x, y, colour) {
  ctx.fillStyle = colour;
  ctx.font = '600 13px Inter, system-ui, sans-serif';
  ctx.fillText(text, x, y);
}

function makeNoise(n) {
  let seed = 42;
  const arr = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    arr.push((seed / 4294967296) * 2 - 1);
  }
  return arr;
}

function generateSignal() {
  const n = 256;
  const fa = Number(freqASlider.value);
  const fb = Number(freqBSlider.value);
  const noise = Number(noiseSlider.value);
  const y = [];
  for (let i = 0; i < n; i++) {
    const t = i / n;
    y.push(
      0.75 * Math.sin(2 * Math.PI * fa * t) +
      0.45 * Math.sin(2 * Math.PI * fb * t + 0.7) +
      noise * signalSeed[i]
    );
  }
  return y;
}

function dftMagnitude(signal, maxFreq) {
  const n = signal.length;
  const mags = [];
  for (let k = 1; k <= maxFreq; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = -2 * Math.PI * k * t / n;
      re += signal[t] * Math.cos(angle);
      im += signal[t] * Math.sin(angle);
    }
    mags.push(Math.sqrt(re * re + im * im) / n);
  }
  return mags;
}

function drawSignalAndSpectrum() {
  const signal = generateSignal();
  drawSignal(signal);
  drawSpectrum(dftMagnitude(signal, 12));
}

function drawSignal(signal) {
  const ctx = setupCanvas(signalCanvas);
  const { w, h } = cssSize(signalCanvas);
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;
  const amp = h * 0.28;

  drawPlotFrame(ctx, w, h, 'A signal made from two hidden rhythms');
  ctx.strokeStyle = 'rgba(30,91,98,0.95)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  signal.forEach((v, i) => {
    const x = 36 + i * ((w - 72) / (signal.length - 1));
    const y = mid - amp * v;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  label(ctx, 'time', w - 62, h - 18, 'rgba(27,31,36,0.62)');
}

function drawSpectrum(mags) {
  const ctx = setupCanvas(spectrumCanvas);
  const { w, h } = cssSize(spectrumCanvas);
  ctx.clearRect(0, 0, w, h);
  drawPlotFrame(ctx, w, h, 'Fourier spectrum: which rotations matched?');

  const maxMag = Math.max(...mags, 0.01);
  const left = 46;
  const bottom = h - 42;
  const plotW = w - 88;
  const plotH = h - 100;
  const barW = plotW / mags.length * 0.62;

  mags.forEach((m, i) => {
    const x = left + i * (plotW / mags.length) + barW * 0.3;
    const barH = (m / maxMag) * plotH;
    ctx.fillStyle = 'rgba(140,84,51,0.86)';
    ctx.fillRect(x, bottom - barH, barW, barH);
    ctx.fillStyle = 'rgba(27,31,36,0.62)';
    ctx.font = '600 12px Inter, system-ui, sans-serif';
    ctx.fillText(String(i + 1), x + barW * 0.28, bottom + 20);
  });
  label(ctx, 'frequency', w - 92, h - 18, 'rgba(27,31,36,0.62)');
}

function drawPlotFrame(ctx, w, h, title) {
  const left = 36;
  const bottom = h - 42;
  ctx.strokeStyle = 'rgba(27,31,36,0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, 42);
  ctx.lineTo(left, bottom);
  ctx.lineTo(w - 36, bottom);
  ctx.stroke();
  label(ctx, title, left, 24, 'rgba(27,31,36,0.76)');
}

function updateReadouts() {
  speedReadout.textContent = `${Number(speedSlider.value).toFixed(1)}×`;
  freqAReadout.textContent = `${freqASlider.value} Hz`;
  freqBReadout.textContent = `${freqBSlider.value} Hz`;
  noiseReadout.textContent = Number(noiseSlider.value).toFixed(2);
}

function animate(time) {
  if (lastTime == null) lastTime = time;
  const dt = (time - lastTime) / 1000;
  lastTime = time;

  if (!paused) {
    theta += dt * Number(speedSlider.value) * 2.2;
  }

  drawCircle();
  drawWaves();
  drawSignalAndSpectrum();
  requestAnimationFrame(animate);
}

speedSlider.addEventListener('input', updateReadouts);
freqASlider.addEventListener('input', () => { updateReadouts(); drawSignalAndSpectrum(); });
freqBSlider.addEventListener('input', () => { updateReadouts(); drawSignalAndSpectrum(); });
noiseSlider.addEventListener('input', () => { updateReadouts(); drawSignalAndSpectrum(); });

pauseButton.addEventListener('click', () => {
  paused = !paused;
  pauseButton.textContent = paused ? 'Play' : 'Pause';
});

window.addEventListener('resize', () => {
  drawCircle();
  drawWaves();
  drawSignalAndSpectrum();
});

updateReadouts();
requestAnimationFrame(animate);

/* ============================================================
   Extra Fourier teaching animations:
   1. Winding the signal around a circle
   2. Signal reconstruction from Fourier components

   Safe to append to fourier-unit-circle.js
   ============================================================ */

(function () {
  const windingCanvas = document.getElementById("windingCanvas");
  const windingSpectrumCanvas = document.getElementById("windingSpectrumCanvas");
  const windingFreqSlider = document.getElementById("windingFreqSlider");
  const windingFreqReadout = document.getElementById("windingFreqReadout");

  const reconstructionCanvas = document.getElementById("reconstructionCanvas");
  const reconstructionSlider = document.getElementById("reconstructionSlider");
  const reconstructionReadout = document.getElementById("reconstructionReadout");

  function setupCanvas(canvas) {
    if (!canvas) return null;

    const ctx = canvas.getContext("2d");

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener("resize", resize);

    return ctx;
  }

  const windingCtx = setupCanvas(windingCanvas);
  const windingSpectrumCtx = setupCanvas(windingSpectrumCanvas);
  const reconstructionCtx = setupCanvas(reconstructionCanvas);

  function getCanvasSize(canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
      w: rect.width,
      h: rect.height
    };
  }

  function signalAt(t) {
    /*
      Teaching signal:
      A mixture of a strong 3 Hz rhythm and a smaller 7 Hz rhythm.
      t is in seconds over a 1-second window.
    */
    return (
      1.0 * Math.sin(2 * Math.PI * 3 * t) +
      0.45 * Math.sin(2 * Math.PI * 7 * t + 0.7)
    );
  }

  function drawAxes(ctx, w, h, x0, y0, x1, y1) {
    ctx.strokeStyle = "rgba(40, 50, 45, 0.25)";
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y0);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0, y1);
    ctx.stroke();
  }

  function drawWinding() {
    if (!windingCanvas || !windingCtx || !windingFreqSlider) return;

    const ctx = windingCtx;
    const { w, h } = getCanvasSize(windingCanvas);

    const testFreq = parseFloat(windingFreqSlider.value);
    if (windingFreqReadout) {
      windingFreqReadout.textContent = `${testFreq.toFixed(1)} Hz`;
    }

    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    const baseR = Math.min(w, h) * 0.24;
    const scale = Math.min(w, h) * 0.10;

    // Background circle
    ctx.strokeStyle = "rgba(40, 50, 45, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR, 0, 2 * Math.PI);
    ctx.stroke();

    const n = 400;
    let sx = 0;
    let sy = 0;

    ctx.beginPath();

    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const x = signalAt(t);

      // Winding: rotate signal around the circle at the test frequency.
      const theta = -2 * Math.PI * testFreq * t;
      const r = baseR + scale * x;

      const px = cx + r * Math.cos(theta);
      const py = cy + r * Math.sin(theta);

      sx += px;
      sy += py;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }

    ctx.strokeStyle = "rgba(39, 76, 94, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();

    const mx = sx / n;
    const my = sy / n;

    // Centre of mass marker
    ctx.fillStyle = "rgba(180, 80, 70, 0.95)";
    ctx.beginPath();
    ctx.arc(mx, my, 6, 0, 2 * Math.PI);
    ctx.fill();

    ctx.strokeStyle = "rgba(180, 80, 70, 0.55)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(mx, my);
    ctx.stroke();

    ctx.fillStyle = "rgba(30, 35, 32, 0.75)";
    ctx.font = "14px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText("wrapped signal", 18, 26);
    ctx.fillText("centre of mass", mx + 10, my - 10);
  }

  function fourierStrengthAt(freq) {
    const n = 700;
    let re = 0;
    let im = 0;

    for (let i = 0; i < n; i++) {
      const t = i / n;
      const x = signalAt(t);
      const theta = -2 * Math.PI * freq * t;

      re += x * Math.cos(theta);
      im += x * Math.sin(theta);
    }

    re /= n;
    im /= n;

    return Math.sqrt(re * re + im * im);
  }

  function drawWindingSpectrum() {
    if (!windingSpectrumCanvas || !windingSpectrumCtx || !windingFreqSlider) return;

    const ctx = windingSpectrumCtx;
    const { w, h } = getCanvasSize(windingSpectrumCanvas);

    const selectedFreq = parseFloat(windingFreqSlider.value);

    ctx.clearRect(0, 0, w, h);

    const padL = 42;
    const padR = 18;
    const padT = 24;
    const padB = 38;

    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    drawAxes(ctx, w, h, padL, h - padB, w - padR, padT);

    const maxF = 12;
    const bars = [];

    let maxVal = 0;
    for (let f = 1; f <= maxF; f++) {
      const val = fourierStrengthAt(f);
      bars.push({ f, val });
      maxVal = Math.max(maxVal, val);
    }

    maxVal = maxVal || 1;

    const barGap = 6;
    const barW = plotW / maxF - barGap;

    for (const b of bars) {
      const x = padL + (b.f - 1) * (plotW / maxF) + barGap / 2;
      const barH = (b.val / maxVal) * plotH;
      const y = h - padB - barH;

      const isSelected = Math.abs(b.f - selectedFreq) < 0.5;

      ctx.fillStyle = isSelected
        ? "rgba(180, 80, 70, 0.9)"
        : "rgba(39, 76, 94, 0.75)";

      ctx.fillRect(x, y, barW, barH);

      ctx.fillStyle = "rgba(30, 35, 32, 0.7)";
      ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(String(b.f), x + barW / 2, h - 14);
    }

    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(30, 35, 32, 0.75)";
    ctx.font = "14px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText("winding strength by frequency", padL, 18);
    ctx.fillText("Hz", w - 36, h - 14);
  }

  function drawReconstruction() {
    if (!reconstructionCanvas || !reconstructionCtx || !reconstructionSlider) return;

    const ctx = reconstructionCtx;
    const { w, h } = getCanvasSize(reconstructionCanvas);

    const nComponents = parseInt(reconstructionSlider.value, 10);

    if (reconstructionReadout) {
      reconstructionReadout.textContent = String(nComponents);
    }

    ctx.clearRect(0, 0, w, h);

    const padL = 42;
    const padR = 18;
    const padT = 24;
    const padB = 38;

    const plotW = w - padL - padR;
    const midY = h * 0.52;
    const ampScale = h * 0.20;

    drawAxes(ctx, w, h, padL, midY, w - padR, padT);

    const n = 600;

    // Original signal
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const x = padL + t * plotW;
      const y = midY - signalAt(t) * ampScale;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.strokeStyle = "rgba(30, 35, 32, 0.30)";
    ctx.lineWidth = 3;
    ctx.stroke();

    // Reconstruction using manually known components first,
    // then extra tiny components to illustrate gradual build-up.
    function reconstructedSignal(t) {
      let y = 0;

      if (nComponents >= 1) {
        y += 1.0 * Math.sin(2 * Math.PI * 3 * t);
      }

      if (nComponents >= 2) {
        y += 0.45 * Math.sin(2 * Math.PI * 7 * t + 0.7);
      }

      // Add smaller illustrative harmonics after the true components.
      for (let k = 3; k <= nComponents; k++) {
        y += 0.08 * Math.sin(2 * Math.PI * k * t + k * 0.4);
      }

      return y;
    }

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const x = padL + t * plotW;
      const y = midY - reconstructedSignal(t) * ampScale;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.strokeStyle = "rgba(39, 76, 94, 0.95)";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.fillStyle = "rgba(30, 35, 32, 0.75)";
    ctx.font = "14px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText("grey = original signal", padL, 18);
    ctx.fillText("blue = reconstruction", padL + 170, 18);
    ctx.fillText("time", w - 48, h - 14);
  }

  function drawAllExtraFourierAnimations() {
    drawWinding();
    drawWindingSpectrum();
    drawReconstruction();
  }

  if (windingFreqSlider) {
    windingFreqSlider.addEventListener("input", drawAllExtraFourierAnimations);
  }

  if (reconstructionSlider) {
    reconstructionSlider.addEventListener("input", drawAllExtraFourierAnimations);
  }

  drawAllExtraFourierAnimations();
})();