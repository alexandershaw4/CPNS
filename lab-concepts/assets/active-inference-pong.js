const canvas = document.getElementById("pongCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  speed: document.getElementById("speedSlider"),
  noise: document.getElementById("noiseSlider"),
  goalPrecision: document.getElementById("goalPrecisionSlider"),
  policyPrecision: document.getElementById("policyPrecisionSlider")
};

const outputs = {
  speed: document.getElementById("speedOut"),
  noise: document.getElementById("noiseOut"),
  goalPrecision: document.getElementById("goalPrecisionOut"),
  policyPrecision: document.getElementById("policyPrecisionOut"),
  crossing: document.getElementById("crossingOut"),
  action: document.getElementById("actionOut"),
  g: document.getElementById("gOut"),
  interpretation: document.getElementById("interpretationOut")
};

const pauseButton = document.getElementById("pauseButton");
const resetButton = document.getElementById("resetButton");
const serveButton = document.getElementById("serveButton");

const court = { x: 70, y: 70, w: 620, h: 480 };
const paddle = { x: court.x + court.w - 34, y: court.y + court.h / 2, w: 14, h: 96 };
const ball = { x: court.x + 110, y: court.y + 180, vx: 2.4, vy: 1.25, r: 10 };

let running = true;
let score = { hits: 0, misses: 0 };
let lastAction = "stay";
let lastG = 0;
let lastEvals = [];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function resetBall(randomise = false) {
  ball.x = court.x + 110;
  ball.y = court.y + (randomise ? 70 + Math.random() * (court.h - 140) : 180);
  const speed = Number(controls.speed.value);
  ball.vx = 1.55 * speed;
  ball.vy = (randomise ? (Math.random() * 2 - 1) : 0.55) * speed;
  if (Math.abs(ball.vy) < 0.45) ball.vy = 0.65 * speed;
}

function predictCrossing(noisy = true) {
  let x = ball.x;
  let y = ball.y;
  let vx = ball.vx;
  let vy = ball.vy;
  const targetX = paddle.x - ball.r;
  let steps = 0;

  while (x < targetX && steps < 600) {
    x += vx;
    y += vy;

    if (y < court.y + ball.r) {
      y = court.y + ball.r + ((court.y + ball.r) - y);
      vy *= -1;
    }
    if (y > court.y + court.h - ball.r) {
      y = court.y + court.h - ball.r - (y - (court.y + court.h - ball.r));
      vy *= -1;
    }

    steps++;
  }

  const noise = Number(controls.noise.value);
  const noiseTerm = noisy ? Math.sin(performance.now() * 0.002 + ball.x * 0.01) * noise * 20 : 0;
  return clamp(y + noiseTerm, court.y + 10, court.y + court.h - 10);
}

function evaluateActions(crossing) {
  const move = 24;
  const actions = [
    { name: "up", dy: -move },
    { name: "stay", dy: 0 },
    { name: "down", dy: move }
  ];

  const goalPrecision = Number(controls.goalPrecision.value);
  const actionCost = { up: 0.12, stay: 0.02, down: 0.12 };

  return actions.map(a => {
    const nextY = clamp(paddle.y + a.dy, court.y + paddle.h / 2, court.y + court.h - paddle.h / 2);
    const mismatch = (crossing - nextY) / 100;
    const G = goalPrecision * mismatch * mismatch + actionCost[a.name];
    return { ...a, nextY, mismatch, G };
  }).sort((a, b) => a.G - b.G);
}

function softmaxProbabilities(evals) {
  const beta = Number(controls.policyPrecision.value);
  const scores = evals.map(e => -beta * e.G);
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const total = expScores.reduce((a, b) => a + b, 0);
  return expScores.map(s => s / total);
}

function chooseAction(evals) {
  return evals[0];
}

function updateAgent() {
  const crossing = predictCrossing(true);
  const evals = evaluateActions(crossing);
  const best = chooseAction(evals);

  lastAction = best.name;
  lastG = best.G;
  lastEvals = evals;

  paddle.y = best.nextY;
  paddle.y = clamp(paddle.y, court.y + paddle.h / 2, court.y + court.h - paddle.h / 2);

  return crossing;
}

function updateBall() {
  const speed = Number(controls.speed.value);
  const base = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
  if (base > 0) {
    const desired = 1.95 * speed;
    ball.vx = (ball.vx / base) * desired;
    ball.vy = (ball.vy / base) * desired;
  }

  ball.x += ball.vx;
  ball.y += ball.vy;

  if (ball.y < court.y + ball.r) {
    ball.y = court.y + ball.r;
    ball.vy *= -1;
  }

  if (ball.y > court.y + court.h - ball.r) {
    ball.y = court.y + court.h - ball.r;
    ball.vy *= -1;
  }

  if (ball.x < court.x + ball.r) {
    ball.x = court.x + ball.r;
    ball.vx *= -1;
  }

  const paddleTop = paddle.y - paddle.h / 2;
  const paddleBottom = paddle.y + paddle.h / 2;
  const paddleLeft = paddle.x;
  const paddleRight = paddle.x + paddle.w;

  if (
    ball.x + ball.r >= paddleLeft &&
    ball.x - ball.r <= paddleRight &&
    ball.y >= paddleTop &&
    ball.y <= paddleBottom &&
    ball.vx > 0
  ) {
    ball.x = paddleLeft - ball.r;
    ball.vx *= -1;
    const offset = (ball.y - paddle.y) / (paddle.h / 2);
    ball.vy += offset * 0.9;
    score.hits += 1;
  }

  if (ball.x > court.x + court.w + 30) {
    score.misses += 1;
    resetBall(true);
  }
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

function drawCourt(crossing) {
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 2;
  roundRect(ctx, court.x, court.y, court.w, court.h, 22, true, true);

  ctx.strokeStyle = "rgba(39,76,94,0.15)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(court.x + court.w / 2, court.y + 18);
  ctx.lineTo(court.x + court.w / 2, court.y + court.h - 18);
  ctx.stroke();
  ctx.setLineDash([]);

  drawPredictedPath(crossing);

  ctx.fillStyle = "#274c5e";
  roundRect(ctx, paddle.x, paddle.y - paddle.h / 2, paddle.w, paddle.h, 7, true, false);

  ctx.fillStyle = "#9a6f48";
  ctx.beginPath();
  ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#6b8f71";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(paddle.x - 18, crossing);
  ctx.lineTo(paddle.x + paddle.w + 18, crossing);
  ctx.stroke();

  ctx.fillStyle = "#6b8f71";
  ctx.font = "800 13px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("predicted crossing", paddle.x - 24, crossing - 8);

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 18px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`hits ${score.hits}   misses ${score.misses}`, court.x + 20, court.y + 32);
}

function drawPredictedPath() {
  let x = ball.x;
  let y = ball.y;
  let vx = ball.vx;
  let vy = ball.vy;

  ctx.strokeStyle = "rgba(107,143,113,0.55)";
  ctx.lineWidth = 3;
  ctx.setLineDash([9, 8]);
  ctx.beginPath();
  ctx.moveTo(x, y);

  for (let i = 0; i < 260; i++) {
    x += vx * 2.4;
    y += vy * 2.4;

    if (y < court.y + ball.r) {
      y = court.y + ball.r + ((court.y + ball.r) - y);
      vy *= -1;
    }
    if (y > court.y + court.h - ball.r) {
      y = court.y + court.h - ball.r - (y - (court.y + court.h - ball.r));
      vy *= -1;
    }

    ctx.lineTo(x, y);
    if (x >= paddle.x) break;
  }

  ctx.stroke();
  ctx.setLineDash([]);
}

function drawPolicyPanel() {
  const x = 730;
  const y = 70;
  const w = 290;
  const h = 480;

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, h, 22, true, true);

  ctx.fillStyle = "#1e2528";
  ctx.font = "900 20px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Candidate actions", x + 22, y + 36);

  const probs = softmaxProbabilities(lastEvals);
  const ordered = lastEvals.slice().sort((a, b) => {
    const order = { up: 0, stay: 1, down: 2 };
    return order[a.name] - order[b.name];
  });

  ordered.forEach((e, i) => {
    const yy = y + 70 + i * 95;
    const isBest = e.name === lastAction;
    const prob = probs[lastEvals.findIndex(v => v.name === e.name)];

    ctx.fillStyle = isBest ? "rgba(107,143,113,0.16)" : "rgba(238,242,240,0.75)";
    roundRect(ctx, x + 20, yy, w - 40, 78, 16, true, false);

    ctx.fillStyle = isBest ? "#6b8f71" : "#274c5e";
    ctx.font = "900 18px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(e.name, x + 38, yy + 32);

    ctx.fillStyle = "#617074";
    ctx.font = "700 12px system-ui, sans-serif";
    ctx.fillText("G", x + 126, yy + 22);
    ctx.fillText("error", x + 176, yy + 22);
    ctx.fillText("P", x + 230, yy + 22);

    ctx.fillStyle = "#1e2528";
    ctx.font = "800 14px system-ui, sans-serif";
    ctx.fillText(e.G.toFixed(2), x + 126, yy + 46);
    ctx.fillText(e.mismatch.toFixed(2), x + 176, yy + 46);
    ctx.fillText(prob.toFixed(2), x + 230, yy + 46);
  });

  ctx.fillStyle = "#617074";
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "left";
  wrapText("The selected action is the policy with lowest expected free energy: low future mismatch plus low movement cost.", x + 22, y + 385, w - 44, 20);
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

function updateOutputs(crossing) {
  outputs.speed.textContent = Number(controls.speed.value).toFixed(1);
  outputs.noise.textContent = Number(controls.noise.value).toFixed(2);
  outputs.goalPrecision.textContent = Number(controls.goalPrecision.value).toFixed(1);
  outputs.policyPrecision.textContent = Number(controls.policyPrecision.value).toFixed(1);
  outputs.crossing.textContent = Math.round(crossing - court.y).toString();
  outputs.action.textContent = lastAction;
  outputs.g.textContent = lastG.toFixed(2);

  const diff = crossing - paddle.y;
  if (Math.abs(diff) < 18) {
    outputs.interpretation.textContent = "The paddle is aligned with the predicted crossing point.";
  } else if (lastAction === "stay") {
    outputs.interpretation.textContent = "Staying is currently expected to produce the least future mismatch.";
  } else {
    outputs.interpretation.textContent = `The paddle moves ${lastAction} toward the predicted crossing point.`;
  }
}

function draw() {
  const crossing = predictCrossing(true);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#1e2528";
  ctx.font = "900 22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Active inference as embodied prediction and policy selection", 70, 36);

  drawCourt(crossing);
  drawPolicyPanel();
  updateOutputs(crossing);
}

function frame() {
  if (running) {
    const crossing = updateAgent();
    updateBall();
  }
  if (lastEvals.length === 0) {
    const crossing = predictCrossing(true);
    lastEvals = evaluateActions(crossing);
    const best = chooseAction(lastEvals);
    lastAction = best.name;
    lastG = best.G;
  }
  draw();
  requestAnimationFrame(frame);
}

Object.values(controls).forEach(control => {
  control.addEventListener("input", () => {
    const crossing = predictCrossing(true);
    lastEvals = evaluateActions(crossing);
    const best = chooseAction(lastEvals);
    lastAction = best.name;
    lastG = best.G;
    draw();
  });
});

pauseButton.addEventListener("click", () => {
  running = !running;
  pauseButton.textContent = running ? "Pause" : "Play";
});

resetButton.addEventListener("click", () => {
  score = { hits: 0, misses: 0 };
  paddle.y = court.y + court.h / 2;
  resetBall(false);
  draw();
});

serveButton.addEventListener("click", () => {
  resetBall(true);
  draw();
});

resetBall(false);
frame();
