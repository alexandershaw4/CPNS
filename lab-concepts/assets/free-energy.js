const canvas = document.getElementById("fepCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  sensory: document.getElementById("sensorySlider"),
  prior: document.getElementById("priorSlider"),
  sensoryPrecision: document.getElementById("sensoryPrecisionSlider"),
  priorPrecision: document.getElementById("priorPrecisionSlider")
};

const outputs = {
  sensory: document.getElementById("sensoryOut"),
  prior: document.getElementById("priorOut"),
  sensoryPrecision: document.getElementById("sensoryPrecisionOut"),
  priorPrecision: document.getElementById("priorPrecisionOut"),
  belief: document.getElementById("beliefOut"),
  freeEnergy: document.getElementById("freeEnergyOut"),
  interpretation: document.getElementById("interpretationOut")
};

const runButton = document.getElementById("runButton");
const resetButton = document.getElementById("resetButton");

let belief = Number(controls.prior.value);
let running = false;
let path = [belief];

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function xMap(v) {
  const left = 82;
  const right = canvas.width - 82;
  return left + ((v + 3.5) / 7) * (right - left);
}

function yMap(F, minF, maxF) {
  const top = 92;
  const bottom = 400;
  return bottom - ((F - minF) / (maxF - minF + 1e-9)) * (bottom - top);
}

function freeEnergy(mu, sensory, prior, sensoryPrecision, priorPrecision) {
  const sensoryError = sensory - mu;
  const priorError = mu - prior;
  return 0.5 * sensoryPrecision * sensoryError * sensoryError +
         0.5 * priorPrecision * priorError * priorError;
}

function gradient(mu, sensory, prior, sensoryPrecision, priorPrecision) {
  return sensoryPrecision * (mu - sensory) + priorPrecision * (mu - prior);
}

function posteriorOptimum(sensory, prior, sensoryPrecision, priorPrecision) {
  return ((sensoryPrecision * sensory) + (priorPrecision * prior)) /
         (sensoryPrecision + priorPrecision);
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

function drawAxis() {
  const axisY = 440;
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(82, axisY);
  ctx.lineTo(canvas.width - 82, axisY);
  ctx.stroke();

  for (let v = -3; v <= 3; v++) {
    const x = xMap(v);
    ctx.strokeStyle = "#d8dedc";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, axisY - 8);
    ctx.lineTo(x, axisY + 8);
    ctx.stroke();

    ctx.fillStyle = "#617074";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(v), x, axisY + 30);
  }

  ctx.fillStyle = "#617074";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("belief about hidden cause", canvas.width / 2, axisY + 56);
}

function drawVerticalMarker(x, label, colour, y1 = 92, y2 = 440) {
  ctx.strokeStyle = colour;
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x, y1);
  ctx.lineTo(x, y2);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.fillStyle = colour;
  ctx.font = "800 14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y1 - 16);
}

function drawLegend() {
  const items = [
    ["sensory input", "#9a6f48"],
    ["prior expectation", "#274c5e"],
    ["current belief", "#6b8f71"],
    ["free energy minimum", "#1e2528"]
  ];

  let x = 80;
  const y = 54;
  items.forEach(([label, colour]) => {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#1e2528";
    ctx.font = "700 13px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, x + 12, y + 5);

    x += ctx.measureText(label).width + 54;
  });
}

function drawCurve(sensory, prior, sensoryPrecision, priorPrecision) {
  const points = [];
  let minF = Infinity;
  let maxF = -Infinity;

  for (let i = 0; i <= 360; i++) {
    const mu = -3.5 + (7 * i) / 360;
    const F = freeEnergy(mu, sensory, prior, sensoryPrecision, priorPrecision);
    points.push({ mu, F });
    minF = Math.min(minF, F);
    maxF = Math.max(maxF, F);
  }

  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 92 + i * (308 / 4);
    ctx.beginPath();
    ctx.moveTo(82, y);
    ctx.lineTo(canvas.width - 82, y);
    ctx.stroke();
  }

  ctx.beginPath();
  points.forEach((p, i) => {
    const x = xMap(p.mu);
    const y = yMap(p.F, minF, maxF);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#6b8f71";
  ctx.lineWidth = 5;
  ctx.lineJoin = "round";
  ctx.stroke();

  return { minF, maxF };
}

function drawPath(minF, maxF, sensory, prior, sensoryPrecision, priorPrecision) {
  if (path.length < 2) return;

  ctx.beginPath();
  path.forEach((mu, i) => {
    const F = freeEnergy(mu, sensory, prior, sensoryPrecision, priorPrecision);
    const x = xMap(mu);
    const y = yMap(F, minF, maxF);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.strokeStyle = "rgba(39, 76, 94, 0.55)";
  ctx.lineWidth = 3;
  ctx.setLineDash([7, 7]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawReadoutPanel(sensory, prior, sensoryPrecision, priorPrecision, F, optimum) {
  const x = 82;
  const y = 490;
  const w = canvas.width - 164;
  const h = 92;

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 18, true, true);

  const items = [
    ["prediction error", sensory - belief, "#9a6f48"],
    ["complexity cost", belief - prior, "#274c5e"],
    ["optimal compromise", optimum, "#1e2528"]
  ];

  items.forEach(([label, value, colour], i) => {
    const bx = x + 28 + i * (w / 3);
    ctx.fillStyle = "#617074";
    ctx.font = "800 12px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label.toUpperCase(), bx, y + 30);

    ctx.fillStyle = colour;
    ctx.font = "900 25px system-ui, sans-serif";
    ctx.fillText(value.toFixed(2), bx, y + 64);
  });
}

function updateOutputs(sensory, prior, sensoryPrecision, priorPrecision, F, optimum) {
  outputs.sensory.textContent = sensory.toFixed(1);
  outputs.prior.textContent = prior.toFixed(1);
  outputs.sensoryPrecision.textContent = sensoryPrecision.toFixed(2);
  outputs.priorPrecision.textContent = priorPrecision.toFixed(2);
  outputs.belief.textContent = belief.toFixed(2);
  outputs.freeEnergy.textContent = F.toFixed(2);

  if (Math.abs(belief - optimum) < 0.08) {
    outputs.interpretation.textContent = "The belief has reached the free-energy minimum: a compromise between evidence and prior expectation.";
  } else if (sensoryPrecision > priorPrecision * 1.5) {
    outputs.interpretation.textContent = "Sensory evidence is more precise, so the minimum shifts toward the sensory input.";
  } else if (priorPrecision > sensoryPrecision * 1.5) {
    outputs.interpretation.textContent = "The prior is more precise, so the minimum stays closer to the prior expectation.";
  } else {
    outputs.interpretation.textContent = "The belief is pulled between sensory evidence and prior expectation.";
  }
}

function draw() {
  const sensory = Number(controls.sensory.value);
  const prior = Number(controls.prior.value);
  const sensoryPrecision = Number(controls.sensoryPrecision.value);
  const priorPrecision = Number(controls.priorPrecision.value);
  const F = freeEnergy(belief, sensory, prior, sensoryPrecision, priorPrecision);
  const optimum = posteriorOptimum(sensory, prior, sensoryPrecision, priorPrecision);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Free energy is lowest at the best precision-weighted explanation", 72, 32);

  drawLegend();

  const scale = drawCurve(sensory, prior, sensoryPrecision, priorPrecision);
  drawPath(scale.minF, scale.maxF, sensory, prior, sensoryPrecision, priorPrecision);

  drawAxis();

  drawVerticalMarker(xMap(sensory), "sensory input", "#9a6f48");
  drawVerticalMarker(xMap(prior), "prior", "#274c5e");

  const optF = freeEnergy(optimum, sensory, prior, sensoryPrecision, priorPrecision);
  ctx.fillStyle = "#1e2528";
  ctx.beginPath();
  ctx.arc(xMap(optimum), yMap(optF, scale.minF, scale.maxF), 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#6b8f71";
  ctx.beginPath();
  ctx.arc(xMap(belief), yMap(F, scale.minF, scale.maxF), 13, 0, Math.PI * 2);
  ctx.fill();

  drawReadoutPanel(sensory, prior, sensoryPrecision, priorPrecision, F, optimum);
  updateOutputs(sensory, prior, sensoryPrecision, priorPrecision, F, optimum);
}

function step() {
  const sensory = Number(controls.sensory.value);
  const prior = Number(controls.prior.value);
  const sensoryPrecision = Number(controls.sensoryPrecision.value);
  const priorPrecision = Number(controls.priorPrecision.value);

  if (running) {
    const g = gradient(belief, sensory, prior, sensoryPrecision, priorPrecision);
    belief -= 0.018 * g;
    belief = clamp(belief, -3.5, 3.5);
    path.push(belief);
    if (path.length > 170) path.shift();

    const optimum = posteriorOptimum(sensory, prior, sensoryPrecision, priorPrecision);
    if (Math.abs(belief - optimum) < 0.015) {
      running = false;
      runButton.textContent = "Run descent";
    }
  }

  draw();
  requestAnimationFrame(step);
}

Object.values(controls).forEach(control => {
  control.addEventListener("input", () => {
    if (control === controls.prior) {
      belief = Number(controls.prior.value);
      path = [belief];
    }
    draw();
  });
});

runButton.addEventListener("click", () => {
  running = !running;
  runButton.textContent = running ? "Pause descent" : "Run descent";
});

resetButton.addEventListener("click", () => {
  belief = Number(controls.prior.value);
  path = [belief];
  running = false;
  runButton.textContent = "Run descent";
  draw();
});

step();
