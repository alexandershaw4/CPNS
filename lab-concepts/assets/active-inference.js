const canvas = document.getElementById("aiCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  goalValue: document.getElementById("goalValueSlider"),
  infoValue: document.getElementById("infoValueSlider"),
  cost: document.getElementById("costSlider"),
  precision: document.getElementById("precisionSlider")
};

const outputs = {
  goalValue: document.getElementById("goalValueOut"),
  infoValue: document.getElementById("infoValueOut"),
  cost: document.getElementById("costOut"),
  precision: document.getElementById("precisionOut"),
  action: document.getElementById("actionOut"),
  g: document.getElementById("gOut"),
  interpretation: document.getElementById("interpretationOut")
};

const stepButton = document.getElementById("stepButton");
const resetButton = document.getElementById("resetButton");
const toggleInfoButton = document.getElementById("toggleInfoButton");

const gridN = 5;
let agent = { x: 1, y: 3 };
let goal = { x: 4, y: 1 };
let clue = { x: 2, y: 1 };
let visitedClue = false;

const actions = [
  { name: "left", dx: -1, dy: 0 },
  { name: "right", dx: 1, dy: 0 },
  { name: "up", dx: 0, dy: -1 },
  { name: "down", dx: 0, dy: 1 }
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function dist(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function nextPos(action) {
  return {
    x: clamp(agent.x + action.dx, 0, gridN - 1),
    y: clamp(agent.y + action.dy, 0, gridN - 1)
  };
}

function evaluateAction(action) {
  const pos = nextPos(action);
  const goalDistance = dist(pos, goal);
  const currentGoalDistance = dist(agent, goal);
  const goalProgress = currentGoalDistance - goalDistance;

  const clueDistance = dist(pos, clue);
  const currentClueDistance = dist(agent, clue);
  const infoProgress = currentClueDistance - clueDistance;

  const reachesGoal = pos.x === goal.x && pos.y === goal.y ? 1 : 0;
  const reachesClue = pos.x === clue.x && pos.y === clue.y && !visitedClue ? 1 : 0;
  const bumpCost = pos.x === agent.x && pos.y === agent.y ? 0.8 : 0;

  const goalValue = Number(controls.goalValue.value);
  const infoValue = Number(controls.infoValue.value);
  const cost = Number(controls.cost.value);

  const pragmaticValue = goalValue * (goalProgress + 2.0 * reachesGoal);
  const epistemicValue = visitedClue ? 0 : infoValue * (0.65 * infoProgress + 1.5 * reachesClue);
  const actionCost = cost + bumpCost;

  const G = actionCost - pragmaticValue - epistemicValue;

  return {
    action: action.name,
    pos,
    G,
    pragmaticValue,
    epistemicValue,
    actionCost,
    goalProgress,
    infoProgress
  };
}

function softmaxPolicy(evals) {
  const beta = Number(controls.precision.value);
  const scores = evals.map(e => -beta * e.G);
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const total = expScores.reduce((a, b) => a + b, 0);
  return expScores.map(s => s / total);
}

function selectedEvaluation() {
  const evals = actions.map(evaluateAction);
  evals.sort((a, b) => a.G - b.G);
  return evals[0];
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

function drawGrid(best) {
  const x0 = 70;
  const y0 = 100;
  const size = 88;
  const gap = 8;

  ctx.fillStyle = "#1e2528";
  ctx.font = "800 22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("The agent selects the action with lowest expected free energy", x0, 40);

  for (let y = 0; y < gridN; y++) {
    for (let x = 0; x < gridN; x++) {
      const px = x0 + x * (size + gap);
      const py = y0 + y * (size + gap);

      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#d8dedc";
      ctx.lineWidth = 1.5;
      roundRect(ctx, px, py, size, size, 16, true, true);

      if (x === goal.x && y === goal.y) {
        ctx.fillStyle = "rgba(107,143,113,0.18)";
        roundRect(ctx, px + 6, py + 6, size - 12, size - 12, 12, true, false);
        ctx.fillStyle = "#6b8f71";
        ctx.font = "900 16px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GOAL", px + size / 2, py + size / 2 + 5);
      }

      if (x === clue.x && y === clue.y && !visitedClue) {
        ctx.fillStyle = "rgba(154,111,72,0.18)";
        roundRect(ctx, px + 6, py + 6, size - 12, size - 12, 12, true, false);
        ctx.fillStyle = "#9a6f48";
        ctx.font = "900 15px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("CLUE", px + size / 2, py + size / 2 + 5);
      }
    }
  }

  const ax = x0 + agent.x * (size + gap) + size / 2;
  const ay = y0 + agent.y * (size + gap) + size / 2;
  ctx.fillStyle = "#274c5e";
  ctx.beginPath();
  ctx.arc(ax, ay, 24, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.font = "900 15px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("AGENT", ax, ay + 5);

  if (best) {
    const bx = x0 + best.pos.x * (size + gap) + size / 2;
    const by = y0 + best.pos.y * (size + gap) + size / 2;
    drawArrow(ax, ay, bx, by, "#6b8f71", 6);
  }

  drawLegend(x0, y0 + gridN * (size + gap) + 36);
}

function drawArrow(x1, y1, x2, y2, colour, width) {
  if (Math.abs(x1 - x2) + Math.abs(y1 - y2) < 2) return;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const startOffset = 32;
  const endOffset = 32;
  const sx = x1 + startOffset * Math.cos(angle);
  const sy = y1 + startOffset * Math.sin(angle);
  const ex = x2 - endOffset * Math.cos(angle);
  const ey = y2 - endOffset * Math.sin(angle);
  const head = 14;

  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(ex - head * Math.cos(angle - Math.PI / 6), ey - head * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(ex - head * Math.cos(angle + Math.PI / 6), ey - head * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function drawLegend(x, y) {
  const items = [
    ["agent", "#274c5e"],
    ["preferred outcome", "#6b8f71"],
    ["informative clue", "#9a6f48"],
    ["selected policy", "#6b8f71"]
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

function drawPolicyTable(evals, probs) {
  const x = 595;
  const y = 102;
  const w = 375;
  const rowH = 66;

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.strokeStyle = "#d8dedc";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, w, 430, 18, true, true);

  ctx.fillStyle = "#1e2528";
  ctx.font = "900 19px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Policy scores", x + 22, y + 34);

  const sorted = evals.slice().sort((a, b) => a.G - b.G);

  sorted.forEach((e, i) => {
    const yy = y + 62 + i * rowH;
    const probability = probs[actions.findIndex(a => a.name === e.action)];

    ctx.fillStyle = i === 0 ? "rgba(107,143,113,0.14)" : "rgba(238,242,240,0.75)";
    roundRect(ctx, x + 18, yy, w - 36, rowH - 10, 14, true, false);

    ctx.fillStyle = i === 0 ? "#6b8f71" : "#274c5e";
    ctx.font = "900 18px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(e.action, x + 36, yy + 34);

    ctx.fillStyle = "#617074";
    ctx.font = "700 12px system-ui, sans-serif";
    ctx.fillText("G", x + 135, yy + 22);
    ctx.fillText("goal", x + 200, yy + 22);
    ctx.fillText("info", x + 265, yy + 22);
    ctx.fillText("P", x + 320, yy + 22);

    ctx.fillStyle = "#1e2528";
    ctx.font = "800 14px system-ui, sans-serif";
    ctx.fillText(e.G.toFixed(2), x + 135, yy + 43);
    ctx.fillText(e.pragmaticValue.toFixed(2), x + 200, yy + 43);
    ctx.fillText(e.epistemicValue.toFixed(2), x + 265, yy + 43);
    ctx.fillText(probability.toFixed(2), x + 320, yy + 43);
  });

  const explainerY = y + 340;
  ctx.fillStyle = "#617074";
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "left";
  wrapText(
    "Lower G is better. Goal value pulls the agent toward preferred outcomes. Information value pulls the agent toward uncertainty-reducing observations.",
    x + 22,
    explainerY,
    w - 44,
    19
  );
}

function wrapText(text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  let line = "";
  for (let n = 0; n < words.length; n++) {
    const testLine = line + words[n] + " ";
    const metrics = ctx.measureText(testLine);
    if (metrics.width > maxWidth && n > 0) {
      ctx.fillText(line, x, y);
      line = words[n] + " ";
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  ctx.fillText(line, x, y);
}

function updateOutputs(best) {
  outputs.goalValue.textContent = Number(controls.goalValue.value).toFixed(1);
  outputs.infoValue.textContent = Number(controls.infoValue.value).toFixed(1);
  outputs.cost.textContent = Number(controls.cost.value).toFixed(2);
  outputs.precision.textContent = Number(controls.precision.value).toFixed(1);

  outputs.action.textContent = best.action;
  outputs.g.textContent = best.G.toFixed(2);

  if (best.epistemicValue > best.pragmaticValue) {
    outputs.interpretation.textContent = "The selected policy is mainly epistemic: the agent is exploring to reduce uncertainty.";
  } else if (best.pragmaticValue > best.epistemicValue) {
    outputs.interpretation.textContent = "The selected policy is mainly pragmatic: the agent is moving toward the preferred outcome.";
  } else {
    outputs.interpretation.textContent = "The agent favours the policy with the best balance of goal value, information gain and cost.";
  }
}

function draw() {
  const evals = actions.map(evaluateAction);
  const probs = softmaxPolicy(evals);
  const best = evals.slice().sort((a, b) => a.G - b.G)[0];

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(best);
  drawPolicyTable(evals, probs);
  updateOutputs(best);
}

Object.values(controls).forEach(control => {
  control.addEventListener("input", draw);
});

stepButton.addEventListener("click", () => {
  const best = selectedEvaluation();
  agent = { x: best.pos.x, y: best.pos.y };

  if (agent.x === clue.x && agent.y === clue.y) {
    visitedClue = true;
  }

  draw();
});

resetButton.addEventListener("click", () => {
  agent = { x: 1, y: 3 };
  visitedClue = false;
  draw();
});

toggleInfoButton.addEventListener("click", () => {
  if (clue.x === 2 && clue.y === 1) {
    clue = { x: 0, y: 0 };
  } else if (clue.x === 0 && clue.y === 0) {
    clue = { x: 3, y: 3 };
  } else {
    clue = { x: 2, y: 1 };
  }
  visitedClue = false;
  draw();
});

draw();
