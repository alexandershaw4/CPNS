const canvas = document.getElementById("bayesCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  priorMean: document.getElementById("priorMean"),
  priorSd: document.getElementById("priorSd"),
  likeMean: document.getElementById("likeMean"),
  likeSd: document.getElementById("likeSd")
};

const outputs = {
  priorMean: document.getElementById("priorMeanOut"),
  priorSd: document.getElementById("priorSdOut"),
  likeMean: document.getElementById("likeMeanOut"),
  likeSd: document.getElementById("likeSdOut"),
  posteriorMean: document.getElementById("posteriorMeanOut"),
  posteriorSd: document.getElementById("posteriorSdOut"),
  interpretation: document.getElementById("interpretationOut")
};

function gaussian(x, mu, sd) {
  return Math.exp(-0.5 * Math.pow((x - mu) / sd, 2)) / (sd * Math.sqrt(2 * Math.PI));
}

function posteriorGaussian(priorMean, priorSd, likeMean, likeSd) {
  const priorPrec = 1 / (priorSd * priorSd);
  const likePrec = 1 / (likeSd * likeSd);
  const postVar = 1 / (priorPrec + likePrec);
  const postMean = postVar * (priorPrec * priorMean + likePrec * likeMean);
  const postSd = Math.sqrt(postVar);
  return { mean: postMean, sd: postSd, priorPrec, likePrec };
}

function mapX(x) {
  const left = 72;
  const right = canvas.width - 40;
  return left + ((x + 5) / 10) * (right - left);
}

function mapY(y, maxY) {
  const top = 54;
  const bottom = canvas.height - 80;
  return bottom - (y / maxY) * (bottom - top);
}

function drawAxes() {
  const left = 72;
  const right = canvas.width - 40;
  const top = 54;
  const bottom = canvas.height - 80;

  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  for (let x = -4; x <= 4; x += 2) {
    const px = mapX(x);
    ctx.beginPath();
    ctx.moveTo(px, bottom);
    ctx.lineTo(px, bottom + 8);
    ctx.stroke();

    ctx.fillStyle = "#617074";
    ctx.font = "14px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(x), px, bottom + 30);
  }

  ctx.fillStyle = "#617074";
  ctx.textAlign = "center";
  ctx.font = "15px system-ui, sans-serif";
  ctx.fillText("Possible hidden state θ", (left + right) / 2, canvas.height - 22);
}

function drawCurve(mu, sd, maxY, stroke, fill, label, dash = []) {
  const xs = [];
  for (let i = 0; i <= 440; i++) {
    const x = -5 + (10 * i) / 440;
    xs.push({ x, y: gaussian(x, mu, sd) });
  }

  const bottom = canvas.height - 80;

  ctx.save();
  ctx.beginPath();
  xs.forEach((p, i) => {
    const px = mapX(p.x);
    const py = mapY(p.y, maxY);
    if (i === 0) ctx.moveTo(px, bottom);
    ctx.lineTo(px, py);
  });
  ctx.lineTo(mapX(5), bottom);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  xs.forEach((p, i) => {
    const px = mapX(p.x);
    const py = mapY(p.y, maxY);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 4;
  ctx.setLineDash(dash);
  ctx.lineJoin = "round";
  ctx.stroke();
  ctx.restore();

  const peakX = mapX(mu);
  const peakY = mapY(gaussian(mu, mu, sd), maxY);

  ctx.fillStyle = stroke;
  ctx.beginPath();
  ctx.arc(peakX, peakY, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = stroke;
  ctx.font = "700 15px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, peakX, Math.max(24, peakY - 16));
}

function drawPrecisionBars(priorPrec, likePrec) {
  const x = 92;
  const y = 70;
  const w = 220;
  const h = 12;
  const total = priorPrec + likePrec;
  const priorW = w * priorPrec / total;
  const likeW = w * likePrec / total;

  ctx.fillStyle = "rgba(255,255,255,0.82)";
  roundRect(ctx, x - 16, y - 42, w + 32, 96, 14, true, false);

  ctx.fillStyle = "#1e2528";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Relative precision", x, y - 18);

  ctx.fillStyle = "rgba(39,76,94,0.88)";
  roundRect(ctx, x, y, priorW, h, 7, true, false);

  ctx.fillStyle = "rgba(154,111,72,0.88)";
  roundRect(ctx, x + priorW, y, likeW, h, 7, true, false);

  ctx.fillStyle = "#617074";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("prior", x, y + 34);
  ctx.fillText("evidence", x + 76, y + 34);
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
  if (width < 0) {
    x += width;
    width = Math.abs(width);
  }
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

function updateOutputs(priorMean, priorSd, likeMean, likeSd, post) {
  outputs.priorMean.textContent = priorMean.toFixed(1);
  outputs.priorSd.textContent = priorSd.toFixed(2);
  outputs.likeMean.textContent = likeMean.toFixed(1);
  outputs.likeSd.textContent = likeSd.toFixed(2);
  outputs.posteriorMean.textContent = post.mean.toFixed(2);
  outputs.posteriorSd.textContent = post.sd.toFixed(2);

  const ratio = post.likePrec / post.priorPrec;
  let interpretation;
  if (ratio > 1.8) {
    interpretation = "Evidence is more precise than the prior, so the posterior moves toward the evidence.";
  } else if (ratio < 0.55) {
    interpretation = "The prior is more precise than the evidence, so the posterior resists the new data.";
  } else {
    interpretation = "Prior and evidence have similar precision, so the posterior sits between them.";
  }

  outputs.interpretation.textContent = interpretation;
}

function draw() {
  const priorMean = Number(controls.priorMean.value);
  const priorSd = Number(controls.priorSd.value);
  const likeMean = Number(controls.likeMean.value);
  const likeSd = Number(controls.likeSd.value);
  const post = posteriorGaussian(priorMean, priorSd, likeMean, likeSd);

  updateOutputs(priorMean, priorSd, likeMean, likeSd, post);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const maxY = Math.max(
    gaussian(priorMean, priorMean, priorSd),
    gaussian(likeMean, likeMean, likeSd),
    gaussian(post.mean, post.mean, post.sd)
  ) * 1.28;

  drawAxes();
  drawPrecisionBars(post.priorPrec, post.likePrec);

  drawCurve(priorMean, priorSd, maxY, "#274c5e", "rgba(39,76,94,0.12)", "prior", [8, 8]);
  drawCurve(likeMean, likeSd, maxY, "#9a6f48", "rgba(154,111,72,0.12)", "evidence", [2, 7]);
  drawCurve(post.mean, post.sd, maxY, "#6b8f71", "rgba(107,143,113,0.18)", "posterior", []);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 18px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Posterior = precision-weighted compromise", 72, 38);
}

Object.values(controls).forEach(control => control.addEventListener("input", draw));

draw();
