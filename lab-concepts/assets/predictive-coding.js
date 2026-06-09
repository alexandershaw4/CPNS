const canvas = document.getElementById("pcCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  trueCause: document.getElementById("trueCause"),
  prediction: document.getElementById("prediction"),
  precision: document.getElementById("precision"),
  learningRate: document.getElementById("learningRate")
};

const outputs = {
  trueCause: document.getElementById("trueCauseOut"),
  prediction: document.getElementById("predictionOut"),
  precision: document.getElementById("precisionOut"),
  learningRate: document.getElementById("learningRateOut"),
  error: document.getElementById("errorOut"),
  weightedError: document.getElementById("weightedErrorOut"),
  interpretation: document.getElementById("interpretationOut")
};

const playPause = document.getElementById("playPause");
const resetButton = document.getElementById("reset");

let belief = Number(controls.prediction.value);
let running = true;
let trace = [belief];

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function xMap(v) {
  const left = 90;
  const right = canvas.width - 90;
  return left + ((v + 3.5) / 7) * (right - left);
}

function yMap(v, top, height) {
  return top + height / 2 - (v / 3.5) * (height * 0.38);
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

function drawArrow(x1, y1, x2, y2, colour, width = 3) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 12;
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function drawScale(y, label) {
  const left = 90;
  const right = canvas.width - 90;

  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(left, y);
  ctx.lineTo(right, y);
  ctx.stroke();

  for (let v = -3; v <= 3; v++) {
    const x = xMap(v);
    ctx.strokeStyle = "#d8dedc";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x, y + 8);
    ctx.stroke();

    ctx.fillStyle = "#617074";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(v), x, y + 30);
  }

  ctx.fillStyle = "#617074";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, (left + right) / 2, y + 56);
}

function drawLegendItem(x, y, colour, label) {
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#1e2528";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, x + 14, y + 5);
}

function drawTracePanel() {
  const x = 90;
  const y = 450;
  const w = canvas.width - 180;
  const h = 120;

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 18, true, true);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 16px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Belief update over time", x + 22, y + 30);

  const midY = y + 72;
  ctx.strokeStyle = "#d8dedc";
  ctx.beginPath();
  ctx.moveTo(x + 22, midY);
  ctx.lineTo(x + w - 22, midY);
  ctx.stroke();

  if (trace.length > 1) {
    ctx.beginPath();
    trace.forEach((v, i) => {
      const px = x + 22 + (i / (trace.length - 1)) * (w - 44);
      const py = yMap(v, y + 26, 84);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "#6b8f71";
    ctx.lineWidth = 4;
    ctx.stroke();
  }

  ctx.fillStyle = "#617074";
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("current belief", x + w - 22, y + 102);
}

function updateOutputs(trueCause, initialPrediction, precision, lr, error, weightedError) {
  outputs.trueCause.textContent = trueCause.toFixed(1);
  outputs.prediction.textContent = initialPrediction.toFixed(1);
  outputs.precision.textContent = precision.toFixed(2);
  outputs.learningRate.textContent = lr.toFixed(2);
  outputs.error.textContent = error.toFixed(2);
  outputs.weightedError.textContent = weightedError.toFixed(2);

  if (Math.abs(error) < 0.08) {
    outputs.interpretation.textContent = "Prediction and sensory input are aligned, so there is little prediction error left to reduce.";
  } else if (precision > 1.6) {
    outputs.interpretation.textContent = "High precision makes the error more influential, so belief updates are stronger.";
  } else if (precision < 0.6) {
    outputs.interpretation.textContent = "Low precision dampens the error, so the system updates cautiously.";
  } else {
    outputs.interpretation.textContent = "The belief moves toward the sensory input in proportion to the weighted prediction error.";
  }
}

function draw(trueCause, initialPrediction, precision, error, weightedError) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Prediction error is the distance between predicted and observed input", 70, 48);

  drawLegendItem(80, 82, "#274c5e", "current prediction / belief");
  drawLegendItem(310, 82, "#9a6f48", "sensory input");
  drawLegendItem(500, 82, "#6b8f71", "update direction");

  const axisY = 230;
  drawScale(axisY, "possible sensory value");

  const beliefX = xMap(belief);
  const causeX = xMap(trueCause);

  ctx.strokeStyle = "rgba(154,111,72,0.45)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(causeX, axisY - 80);
  ctx.lineTo(causeX, axisY + 80);
  ctx.stroke();

  ctx.fillStyle = "#9a6f48";
  ctx.beginPath();
  ctx.arc(causeX, axisY, 12, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#274c5e";
  ctx.beginPath();
  ctx.arc(beliefX, axisY, 12, 0, Math.PI * 2);
  ctx.fill();

  const arrowWidth = clamp(Math.abs(weightedError) * 1.4, 2.5, 10);
  drawArrow(beliefX, axisY - 48, causeX, axisY - 48, "#6b8f71", arrowWidth);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 15px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("prediction error", (beliefX + causeX) / 2, axisY - 68);

  const panelY = 330;
  const panelW = 268;
  const panelH = 76;
  const gap = 18;
  const startX = 90;

  const panels = [
    ["Prediction", belief.toFixed(2), "#274c5e"],
    ["Sensory input", trueCause.toFixed(2), "#9a6f48"],
    ["Weighted error", weightedError.toFixed(2), "#6b8f71"]
  ];

  panels.forEach((p, i) => {
    const x = startX + i * (panelW + gap);
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.strokeStyle = "#d8dedc";
    ctx.lineWidth = 1;
    roundRect(ctx, x, panelY, panelW, panelH, 16, true, true);

    ctx.fillStyle = "#617074";
    ctx.font = "800 12px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(p[0].toUpperCase(), x + 18, panelY + 28);

    ctx.fillStyle = p[2];
    ctx.font = "900 26px system-ui, sans-serif";
    ctx.fillText(p[1], x + 18, panelY + 58);
  });

  drawTracePanel();
}

function animate() {
  const trueCause = Number(controls.trueCause.value);
  const initialPrediction = Number(controls.prediction.value);
  const precision = Number(controls.precision.value);
  const lr = Number(controls.learningRate.value);

  const error = trueCause - belief;
  const weightedError = precision * error;

  if (running) {
    belief += lr * weightedError;
    belief = clamp(belief, -3.5, 3.5);
    trace.push(belief);
    if (trace.length > 160) trace.shift();
  }

  updateOutputs(trueCause, initialPrediction, precision, lr, error, weightedError);
  draw(trueCause, initialPrediction, precision, error, weightedError);

  requestAnimationFrame(animate);
}

Object.values(controls).forEach(control => {
  control.addEventListener("input", () => {
    if (control === controls.prediction) {
      belief = Number(controls.prediction.value);
      trace = [belief];
    }
  });
});

playPause.addEventListener("click", () => {
  running = !running;
  playPause.textContent = running ? "Pause" : "Play";
});

resetButton.addEventListener("click", () => {
  belief = Number(controls.prediction.value);
  trace = [belief];
});

animate();
