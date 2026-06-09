const canvas = document.getElementById("pebCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  effect: document.getElementById("effectSlider"),
  uncertainty: document.getElementById("uncertaintySlider"),
  variation: document.getElementById("variationSlider"),
  n: document.getElementById("nSlider")
};

const outputs = {
  effect: document.getElementById("effectOut"),
  uncertainty: document.getElementById("uncertaintyOut"),
  variation: document.getElementById("variationOut"),
  n: document.getElementById("nOut"),
  beta: document.getElementById("betaOut"),
  pp: document.getElementById("ppOut"),
  interpretation: document.getElementById("interpretationOut")
};

const resampleButton = document.getElementById("resampleButton");
const toggleShrinkageButton = document.getElementById("toggleShrinkageButton");

let seed = 7;
let showShrinkage = true;
let subjects = [];

function rng() {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function generateSubjects() {
  const n = Number(controls.n.value);
  const effect = Number(controls.effect.value);
  const uncertainty = Number(controls.uncertainty.value);
  const variation = Number(controls.variation.value);

  subjects = [];
  for (let i = 0; i < n; i++) {
    const group = i < n / 2 ? 0 : 1;
    const groupMean = group === 0 ? -effect / 2 : effect / 2;
    const trueTheta = groupMean + randn() * variation;
    const se = uncertainty * (0.55 + rng() * 0.9);
    const estimate = trueTheta + randn() * se;
    subjects.push({ i, group, trueTheta, estimate, se });
  }
}

function analyseSubjects() {
  const g0 = subjects.filter(s => s.group === 0);
  const g1 = subjects.filter(s => s.group === 1);

  const weightedMean = arr => {
    let sw = 0, swx = 0;
    arr.forEach(s => {
      const w = 1 / (s.se * s.se + 1e-6);
      sw += w;
      swx += w * s.estimate;
    });
    return swx / sw;
  };

  const m0 = weightedMean(g0);
  const m1 = weightedMean(g1);
  const beta = m1 - m0;

  let seBeta = Math.sqrt(
    1 / g0.reduce((a, s) => a + 1 / (s.se * s.se + 1e-6), 0) +
    1 / g1.reduce((a, s) => a + 1 / (s.se * s.se + 1e-6), 0)
  );

  const variation = Number(controls.variation.value);
  seBeta = Math.sqrt(seBeta * seBeta + variation * variation / Math.max(8, subjects.length));

  const pp = beta >= 0 ? normalCdf(beta / seBeta) : normalCdf(-beta / seBeta);
  return { m0, m1, beta, seBeta, pp };
}

function shrinkSubject(s, stats) {
  const groupMean = s.group === 0 ? stats.m0 : stats.m1;
  const groupVar = Math.pow(Number(controls.variation.value), 2) + 0.04;
  const subjVar = s.se * s.se;
  const w = groupVar / (groupVar + subjVar);
  return w * s.estimate + (1 - w) * groupMean;
}

function xMap(index, n) {
  const left = 82;
  const right = 675;
  return left + (index / Math.max(1, n - 1)) * (right - left);
}

function yMap(value) {
  const top = 82;
  const bottom = 510;
  const minY = -2.3;
  const maxY = 2.3;
  return bottom - ((value - minY) / (maxY - minY)) * (bottom - top);
}

function roundRect(ctx, x, y, w, h, r, fill, stroke) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

function drawAxes() {
  const left = 82;
  const right = 675;
  const top = 82;
  const bottom = 510;

  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;

  for (let y = -2; y <= 2; y += 1) {
    const py = yMap(y);
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(right, py);
    ctx.stroke();

    ctx.fillStyle = "#617074";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(y.toFixed(0), left - 10, py + 5);
  }

  ctx.strokeStyle = "#1e2528";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.stroke();

  ctx.fillStyle = "#617074";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("subjects", (left + right) / 2, 552);

  ctx.save();
  ctx.translate(28, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("DCM parameter estimate", 0, 0);
  ctx.restore();
}

function drawLegend() {
  const x = 92;
  const y = 44;
  const items = [
    ["group A", "#274c5e"],
    ["group B", "#9a6f48"],
    ["PEB shrinkage estimate", "#6b8f71"]
  ];

  let xx = x;
  items.forEach(([label, colour]) => {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(xx, y, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#1e2528";
    ctx.font = "700 13px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, xx + 12, y + 5);
    xx += ctx.measureText(label).width + 54;
  });
}

function drawSubjects(stats) {
  const n = subjects.length;

  subjects.forEach((s, idx) => {
    const x = xMap(idx, n);
    const colour = s.group === 0 ? "#274c5e" : "#9a6f48";

    ctx.strokeStyle = "rgba(97,112,116,0.55)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, yMap(s.estimate - 1.96 * s.se));
    ctx.lineTo(x, yMap(s.estimate + 1.96 * s.se));
    ctx.stroke();

    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(x, yMap(s.estimate), 6, 0, Math.PI * 2);
    ctx.fill();

    if (showShrinkage) {
      const shrunk = shrinkSubject(s, stats);

      ctx.strokeStyle = "rgba(107,143,113,0.35)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, yMap(s.estimate));
      ctx.lineTo(x, yMap(shrunk));
      ctx.stroke();

      ctx.fillStyle = "#6b8f71";
      ctx.beginPath();
      ctx.arc(x, yMap(shrunk), 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  const splitX = xMap(Math.floor(n / 2) - 0.5, n);
  ctx.strokeStyle = "rgba(30,37,40,0.18)";
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(splitX, 82);
  ctx.lineTo(splitX, 510);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#617074";
  ctx.font = "800 14px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Group A", xMap(Math.floor(n * 0.25), n), 535);
  ctx.fillText("Group B", xMap(Math.floor(n * 0.75), n), 535);
}

function drawGroupMeans(stats) {
  const n = subjects.length;
  const x0 = xMap(Math.floor(n * 0.25), n);
  const x1 = xMap(Math.floor(n * 0.75), n);

  ctx.strokeStyle = "#1e2528";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(x0 - 44, yMap(stats.m0));
  ctx.lineTo(x0 + 44, yMap(stats.m0));
  ctx.moveTo(x1 - 44, yMap(stats.m1));
  ctx.lineTo(x1 + 44, yMap(stats.m1));
  ctx.stroke();

  ctx.strokeStyle = "#6b8f71";
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(x0 + 48, yMap(stats.m0));
  ctx.lineTo(x1 - 48, yMap(stats.m1));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("group effect β", (x0 + x1) / 2, (yMap(stats.m0) + yMap(stats.m1)) / 2 - 12);
}

function drawSidePanel(stats) {
  const x = 725;
  const y = 82;
  const w = 285;
  const h = 428;

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 22, true, true);

  ctx.fillStyle = "#1e2528";
  ctx.font = "900 20px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("PEB summary", x + 22, y + 36);

  const items = [
    ["Group A mean", stats.m0],
    ["Group B mean", stats.m1],
    ["Effect β", stats.beta],
    ["Uncertainty SD", stats.seBeta],
    ["P(effect sign)", stats.pp]
  ];

  items.forEach(([label, value], i) => {
    const yy = y + 78 + i * 58;
    ctx.fillStyle = "rgba(238,242,240,0.78)";
    roundRect(ctx, x + 20, yy, w - 40, 44, 12, true, false);

    ctx.fillStyle = "#617074";
    ctx.font = "800 12px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label.toUpperCase(), x + 34, yy + 18);

    ctx.fillStyle = label.includes("Effect") ? "#6b8f71" : "#1e2528";
    ctx.font = "900 18px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(2), x + w - 34, yy + 29);
  });

  ctx.fillStyle = "#617074";
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "left";
  wrapText("Dots are first-level estimates. Bars are uncertainty. Green dots show empirical-Bayes shrinkage toward the group model.", x + 22, y + 365, w - 44, 19);
}

function wrapText(text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  let line = "";
  for (let n = 0; n < words.length; n++) {
    const testLine = line + words[n] + " ";
    if (ctx.measureText(testLine).width > maxWidth && n > 0) {
      ctx.fillText(line, x, y);
      line = words[n] + " ";
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  ctx.fillText(line, x, y);
}

function updateOutputs(stats) {
  outputs.effect.textContent = Number(controls.effect.value).toFixed(2);
  outputs.uncertainty.textContent = Number(controls.uncertainty.value).toFixed(2);
  outputs.variation.textContent = Number(controls.variation.value).toFixed(2);
  outputs.n.textContent = String(Number(controls.n.value));

  outputs.beta.textContent = stats.beta.toFixed(2);
  outputs.pp.textContent = stats.pp.toFixed(2);

  if (stats.pp > 0.95) {
    outputs.interpretation.textContent = "There is high posterior confidence in the direction of the group effect.";
  } else if (stats.pp > 0.8) {
    outputs.interpretation.textContent = "There is moderate posterior confidence in the direction of the group effect.";
  } else {
    outputs.interpretation.textContent = "The group effect is uncertain because the estimates overlap substantially relative to their uncertainty.";
  }
}

function draw() {
  const stats = analyseSubjects();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#1e2528";
  ctx.font = "900 22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("PEB models subject parameters and their uncertainty at the group level", 70, 32);

  drawLegend();
  drawAxes();
  drawSubjects(stats);
  drawGroupMeans(stats);
  drawSidePanel(stats);
  updateOutputs(stats);
}

Object.values(controls).forEach(control => {
  control.addEventListener("input", () => {
    generateSubjects();
    draw();
  });
});

resampleButton.addEventListener("click", () => {
  seed = Math.floor(Math.random() * 1000000);
  generateSubjects();
  draw();
});

toggleShrinkageButton.addEventListener("click", () => {
  showShrinkage = !showShrinkage;
  toggleShrinkageButton.textContent = showShrinkage ? "Hide shrinkage" : "Show shrinkage";
  draw();
});

generateSubjects();
draw();
