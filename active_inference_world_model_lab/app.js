/*
  Active Inference World Model Lab
  V1.4: self-contained browser script with animated movement, pulsing routes and search events.
  This means the demo works from file:// as well as from a local web server.
*/

const rooms = [
  {
    id: "kitchen",
    name: "Kitchen",
    surface: "table",
    rect: { x: 8, y: 10, w: 38, h: 38 },
    agent: { x: 27, y: 32 },
    mug: { x: 35, y: 29 },
    surfacePos: { x: 52, y: 58 }
  },
  {
    id: "office",
    name: "Office",
    surface: "desk",
    rect: { x: 54, y: 10, w: 38, h: 38 },
    agent: { x: 73, y: 32 },
    mug: { x: 80, y: 29 },
    surfacePos: { x: 52, y: 58 }
  },
  {
    id: "living_room",
    name: "Living room",
    surface: "sofa",
    rect: { x: 8, y: 56, w: 38, h: 34 },
    agent: { x: 27, y: 73 },
    mug: { x: 35, y: 73 },
    surfacePos: { x: 52, y: 56 }
  },
  {
    id: "bedroom",
    name: "Bedroom",
    surface: "bed",
    rect: { x: 54, y: 56, w: 38, h: 34 },
    agent: { x: 73, y: 73 },
    mug: { x: 80, y: 73 },
    surfacePos: { x: 52, y: 56 }
  }
];

const roomLookup = Object.fromEntries(rooms.map(room => [room.id, room]));
const ROOM_IDS = rooms.map(room => room.id);
const EPS = 1e-9;

const graph = {
  kitchen: ["living_room", "office"],
  office: ["kitchen", "living_room", "bedroom"],
  living_room: ["kitchen", "office", "bedroom"],
  bedroom: ["living_room", "office"]
};

const routeLines = [
  ["kitchen", "office"],
  ["kitchen", "living_room"],
  ["office", "bedroom"],
  ["living_room", "bedroom"],
  ["living_room", "office"]
];

const objectPriors = {
  mug: {
    kitchen: 0.62,
    office: 0.24,
    living_room: 0.10,
    bedroom: 0.04
  }
};

const presetSettings = {
  default: {
    label: "Balanced agent",
    semanticStrength: 1.00,
    sensoryPrecision: 0.86,
    epistemicWeight: 1.20,
    goalPreference: 2.20,
    actionCost: 0.35,
    decisionNoise: 0.00
  },
  flat: {
    label: "Flat priors",
    semanticStrength: 0.00,
    sensoryPrecision: 0.86,
    epistemicWeight: 1.20,
    goalPreference: 2.20,
    actionCost: 0.35,
    decisionNoise: 0.00
  },
  curious: {
    label: "Curious explorer",
    semanticStrength: 0.65,
    sensoryPrecision: 0.86,
    epistemicWeight: 2.65,
    goalPreference: 1.10,
    actionCost: 0.20,
    decisionNoise: 0.00
  },
  noisy: {
    label: "Noisy senses",
    semanticStrength: 1.00,
    sensoryPrecision: 0.35,
    epistemicWeight: 1.65,
    goalPreference: 2.00,
    actionCost: 0.35,
    decisionNoise: 0.30
  },
  lazy: {
    label: "High action cost",
    semanticStrength: 1.00,
    sensoryPrecision: 0.86,
    epistemicWeight: 0.70,
    goalPreference: 2.20,
    actionCost: 1.35,
    decisionNoise: 0.00
  }
};

function titleCaseRoom(id) {
  return roomLookup[id]?.name ?? id;
}

function graphDistance(start, goal) {
  if (start === goal) return 0;
  const path = shortestPath(start, goal);
  return path.length ? path.length - 1 : 99;
}

function routeKey(a, b) {
  return [a, b].sort().join("__");
}

function shortestPath(start, goal) {
  if (start === goal) return [start];
  const visited = new Set([start]);
  const queue = [[start]];

  while (queue.length > 0) {
    const path = queue.shift();
    const node = path[path.length - 1];
    for (const next of graph[node] ?? []) {
      if (visited.has(next)) continue;
      const nextPath = [...path, next];
      if (next === goal) return nextPath;
      visited.add(next);
      queue.push(nextPath);
    }
  }
  return [];
}

function normalise(distribution) {
  const total = Object.values(distribution).reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    const flat = 1 / rooms.length;
    return Object.fromEntries(rooms.map(room => [room.id, flat]));
  }
  return Object.fromEntries(Object.entries(distribution).map(([key, value]) => [key, value / total]));
}

function entropy(distribution) {
  return Object.values(distribution).reduce((sum, p) => {
    if (p <= 0) return sum;
    return sum - p * Math.log2(p);
  }, 0);
}

function normalisedEntropy(distribution) {
  return entropy(distribution) / Math.log2(Object.keys(distribution).length);
}

function mostLikelyRoom(distribution) {
  return Object.entries(distribution).sort((a, b) => b[1] - a[1])[0][0];
}

function formatProbability(value) {
  return Number(value).toFixed(2);
}

function formatSigned(value) {
  const rounded = Number(value).toFixed(2);
  return Number(value) > 0 ? `+${rounded}` : rounded;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function gaussianNoise() {
  const u = Math.max(Math.random(), EPS);
  const v = Math.max(Math.random(), EPS);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

class ActiveInferenceWorldModel {
  constructor() {
    this.defaultSettings = { ...presetSettings.default };
    this.reset();
  }

  reset(overrides = {}) {
    this.objectId = "mug";
    this.agentLocation = "living_room";
    this.hiddenRoom = overrides.hiddenRoom ?? "office";
    this.settings = { ...this.defaultSettings, ...(overrides.settings ?? {}) };
    this.observations = [];
    this.success = false;
    this.lastAction = null;
    this.trace = [
      "The agent starts in the living room with a semantic prior: mugs are usually found in kitchens, sometimes on desks and rarely in bedrooms."
    ];
    this.recomputeBeliefs();
  }

  setHiddenRoom(roomId) {
    this.hiddenRoom = roomId;
    this.observations = [];
    this.success = false;
    this.lastAction = null;
    this.agentLocation = "living_room";
    this.trace = [
      `The hidden mug location has been reset to ${titleCaseRoom(roomId).toLowerCase()}. The agent does not know this and only sees its own beliefs.`
    ];
    this.recomputeBeliefs();
  }

  updateSettings(partialSettings) {
    this.settings = { ...this.settings, ...partialSettings };
    this.recomputeBeliefs();
  }

  priorDistribution() {
    const semantic = objectPriors[this.objectId];
    const flat = Object.fromEntries(ROOM_IDS.map(id => [id, 1 / ROOM_IDS.length]));
    const strength = this.settings.semanticStrength;

    if (strength <= 1) {
      const blended = {};
      for (const id of ROOM_IDS) {
        blended[id] = (1 - strength) * flat[id] + strength * semantic[id];
      }
      return normalise(blended);
    }

    const sharpened = {};
    for (const id of ROOM_IDS) {
      sharpened[id] = Math.pow(semantic[id] + EPS, strength);
    }
    return normalise(sharpened);
  }

  recomputeBeliefs() {
    let posterior = this.priorDistribution();

    for (const observation of this.observations) {
      posterior = this.applyObservation(posterior, observation);
    }

    this.beliefs = posterior;
    this.uncertainty = normalisedEntropy(posterior);
    this.policies = this.evaluatePolicies();
  }

  applyObservation(prior, observation) {
    const likelihood = this.observationLikelihood(observation.room, observation.detected);
    const unnormalised = {};
    for (const id of ROOM_IDS) {
      unnormalised[id] = prior[id] * likelihood[id];
    }
    return normalise(unnormalised);
  }

  observationLikelihood(roomToSearch, detected) {
    const precision = clamp(this.settings.sensoryPrecision, 0.01, 0.99);
    const miss = 1 - precision;

    const likelihood = {};
    for (const id of ROOM_IDS) {
      const objectIsInSearchedRoom = id === roomToSearch;
      if (detected) {
        likelihood[id] = objectIsInSearchedRoom ? precision : 0.015;
      } else {
        likelihood[id] = objectIsInSearchedRoom ? Math.max(miss, 0.01) : 0.985;
      }
    }
    return likelihood;
  }

  expectedInformationGain(roomToSearch) {
    const precision = clamp(this.settings.sensoryPrecision, 0.01, 0.99);
    const pHere = this.beliefs[roomToSearch];
    const falseAlarm = 0.015;
    const pDetected = pHere * precision + (1 - pHere) * falseAlarm;
    const detectedPosterior = this.applyObservation(this.beliefs, { room: roomToSearch, detected: true });
    const notDetectedPosterior = this.applyObservation(this.beliefs, { room: roomToSearch, detected: false });
    const expectedEntropy = pDetected * entropy(detectedPosterior) + (1 - pDetected) * entropy(notDetectedPosterior);
    return Math.max(0, entropy(this.beliefs) - expectedEntropy);
  }

  evaluatePolicies() {
    const policies = ROOM_IDS.map(roomId => {
      const distance = graphDistance(this.agentLocation, roomId);
      const pTarget = this.beliefs[roomId];
      const infoGain = this.expectedInformationGain(roomId);
      const cost = this.settings.actionCost * distance;
      const goalValue = this.settings.goalPreference * pTarget;
      const epistemicValue = this.settings.epistemicWeight * infoGain;
      const expectedFreeEnergy = cost - goalValue - epistemicValue;

      return {
        roomId,
        title: `Search ${titleCaseRoom(roomId).toLowerCase()}`,
        distance,
        pTarget,
        infoGain,
        cost,
        goalValue,
        epistemicValue,
        expectedFreeEnergy
      };
    });

    return policies.sort((a, b) => a.expectedFreeEnergy - b.expectedFreeEnergy);
  }

  choosePolicy() {
    this.recomputeBeliefs();
    if (this.settings.decisionNoise <= 0) {
      return this.policies[0];
    }

    const noisy = this.policies.map(policy => ({
      ...policy,
      noisyG: policy.expectedFreeEnergy + gaussianNoise() * this.settings.decisionNoise
    }));
    return noisy.sort((a, b) => a.noisyG - b.noisyG)[0];
  }

  step() {
    if (this.success) {
      this.trace.unshift("The mug has already been found. Reset the demo or move the hidden state to run a new search.");
      return this.snapshot();
    }

    const beforeBeliefs = { ...this.beliefs };
    const policy = this.choosePolicy();
    const previousLocation = this.agentLocation;
    this.agentLocation = policy.roomId;

    const searchedCorrectRoom = policy.roomId === this.hiddenRoom;
    const detected = searchedCorrectRoom && Math.random() < this.settings.sensoryPrecision;
    this.observations.push({ room: policy.roomId, detected });
    this.recomputeBeliefs();

    const before = beforeBeliefs[policy.roomId];
    const after = this.beliefs[policy.roomId];
    this.lastAction = {
      from: previousLocation,
      to: policy.roomId,
      searchedRoom: policy.roomId,
      detected,
      expectedFreeEnergy: policy.expectedFreeEnergy
    };

    if (detected) {
      this.success = true;
      this.trace.unshift(
        `The agent moved from ${titleCaseRoom(previousLocation).toLowerCase()} to ${titleCaseRoom(policy.roomId).toLowerCase()} and found the mug. Belief in that location rose from ${formatProbability(before)} to ${formatProbability(after)}.`
      );
      return this.snapshot();
    }

    const revisionDirection = after < before ? "decreased" : "changed";
    const nextBest = titleCaseRoom(mostLikelyRoom(this.beliefs)).toLowerCase();
    this.trace.unshift(
      `The agent searched ${titleCaseRoom(policy.roomId).toLowerCase()} because it had low expected free energy. The mug was not observed, so belief in that location ${revisionDirection} from ${formatProbability(before)} to ${formatProbability(after)}; the next most plausible location is now ${nextBest}.`
    );

    return this.snapshot();
  }

  snapshot() {
    this.recomputeBeliefs();
    return {
      objectId: this.objectId,
      agentLocation: this.agentLocation,
      hiddenRoom: this.hiddenRoom,
      settings: { ...this.settings },
      beliefs: { ...this.beliefs },
      uncertainty: this.uncertainty,
      mostLikely: mostLikelyRoom(this.beliefs),
      observations: [...this.observations],
      success: this.success,
      lastAction: this.lastAction ? { ...this.lastAction } : null,
      policies: this.policies.map(policy => ({ ...policy })),
      trace: [...this.trace],
      prior: this.priorDistribution()
    };
  }
}

class LabUI {
  constructor(model) {
    this.model = model;
    this.revealed = false;
    this.autoTimer = null;
    this.movementRoute = null;
    this.searchTimer = null;
    this.routeTimer = null;
    this.dom = this.collectDom();
    this.renderWorldShell();
    this.bindEvents();
    this.render();
  }

  collectDom() {
    return {
      worldMap: document.querySelector("#worldMap"),
      revealBtn: document.querySelector("#revealBtn"),
      hiddenRoomSelect: document.querySelector("#hiddenRoomSelect"),
      agentLocationLabel: document.querySelector("#agentLocationLabel"),
      beliefBars: document.querySelector("#beliefBars"),
      uncertaintyLabel: document.querySelector("#uncertaintyLabel"),
      mostLikelyLabel: document.querySelector("#mostLikelyLabel"),
      searchCountLabel: document.querySelector("#searchCountLabel"),
      policyTable: document.querySelector("#policyTable"),
      traceList: document.querySelector("#traceList"),
      stepBtn: document.querySelector("#stepBtn"),
      autoBtn: document.querySelector("#autoBtn"),
      resetBtn: document.querySelector("#resetBtn"),
      advancedToggle: document.querySelector("#advancedToggle"),
      advancedPanel: document.querySelector("#advancedPanel"),
      priorReadout: document.querySelector("#priorReadout"),
      likelihoodReadout: document.querySelector("#likelihoodReadout"),
      efeReadout: document.querySelector("#efeReadout"),
      presetButtons: [...document.querySelectorAll("[data-preset]")],
      sliders: {
        semanticStrength: document.querySelector("#semanticStrength"),
        sensoryPrecision: document.querySelector("#sensoryPrecision"),
        epistemicWeight: document.querySelector("#epistemicWeight"),
        goalPreference: document.querySelector("#goalPreference"),
        actionCost: document.querySelector("#actionCost"),
        decisionNoise: document.querySelector("#decisionNoise")
      },
      outputs: {
        semanticStrength: document.querySelector("#semanticStrengthOut"),
        sensoryPrecision: document.querySelector("#sensoryPrecisionOut"),
        epistemicWeight: document.querySelector("#epistemicWeightOut"),
        goalPreference: document.querySelector("#goalPreferenceOut"),
        actionCost: document.querySelector("#actionCostOut"),
        decisionNoise: document.querySelector("#decisionNoiseOut")
      }
    };
  }

  renderWorldShell() {
    const map = this.dom.worldMap;
    map.innerHTML = "";

    for (const [fromId, toId] of routeLines) {
      const from = roomLookup[fromId].agent;
      const to = roomLookup[toId].agent;
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.sqrt(dx * dx + dy * dy);
      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
      const line = document.createElement("div");
      line.className = "route-line";
      line.dataset.routeKey = routeKey(fromId, toId);
      line.style.left = `${from.x}%`;
      line.style.top = `${from.y}%`;
      line.style.width = `${length}%`;
      line.style.transform = `rotate(${angle}deg)`;
      map.appendChild(line);
    }

    for (const room of rooms) {
      const node = document.createElement("div");
      node.className = "room";
      node.dataset.roomId = room.id;
      node.style.left = `${room.rect.x}%`;
      node.style.top = `${room.rect.y}%`;
      node.style.width = `${room.rect.w}%`;
      node.style.height = `${room.rect.h}%`;

      const beliefGlow = document.createElement("div");
      beliefGlow.className = "room-belief-glow";
      node.appendChild(beliefGlow);

      const label = document.createElement("div");
      label.className = "room-label";
      label.textContent = room.name;
      node.appendChild(label);

      const surface = document.createElement("div");
      surface.className = "surface";
      surface.textContent = room.surface;
      surface.style.left = `${room.surfacePos.x}%`;
      surface.style.top = `${room.surfacePos.y}%`;
      node.appendChild(surface);

      const beliefPill = document.createElement("div");
      beliefPill.className = "room-belief-pill";
      beliefPill.dataset.beliefPill = room.id;
      beliefPill.textContent = "0.00";
      node.appendChild(beliefPill);

      const searchBurst = document.createElement("div");
      searchBurst.className = "search-burst";
      searchBurst.dataset.searchBurst = room.id;
      searchBurst.style.left = `${room.surfacePos.x}%`;
      searchBurst.style.top = `${room.surfacePos.y}%`;
      searchBurst.textContent = "search";
      node.appendChild(searchBurst);

      map.appendChild(node);
    }

    const mug = document.createElement("div");
    mug.className = "mug-marker hidden";
    mug.dataset.mugMarker = "true";
    mug.textContent = "☕";
    map.appendChild(mug);

    const agent = document.createElement("div");
    agent.className = "agent";
    agent.dataset.agent = "true";
    map.appendChild(agent);
  }

  bindEvents() {
    this.dom.stepBtn.addEventListener("click", () => {
      this.stopAutoRun();
      this.runStep();
    });

    this.dom.autoBtn.addEventListener("click", () => {
      if (this.autoTimer) {
        this.stopAutoRun();
      } else {
        this.startAutoRun();
      }
    });

    this.dom.resetBtn.addEventListener("click", () => {
      this.stopAutoRun();
      this.clearTransientAnimations();
      this.model.reset({ hiddenRoom: this.dom.hiddenRoomSelect.value, settings: this.getSettingsFromControls() });
      this.render();
    });

    this.dom.revealBtn.addEventListener("click", () => {
      this.revealed = !this.revealed;
      this.dom.revealBtn.textContent = this.revealed ? "Hide hidden state" : "Reveal hidden state";
      this.render();
    });

    this.dom.hiddenRoomSelect.addEventListener("change", () => {
      this.stopAutoRun();
      this.clearTransientAnimations();
      this.model.setHiddenRoom(this.dom.hiddenRoomSelect.value);
      this.render();
    });

    this.dom.advancedToggle.addEventListener("change", () => {
      this.dom.advancedPanel.classList.toggle("hidden", !this.dom.advancedToggle.checked);
      this.render();
    });

    for (const [key, slider] of Object.entries(this.dom.sliders)) {
      slider.addEventListener("input", () => {
        this.setActivePreset(null);
        this.model.updateSettings({ [key]: Number(slider.value) });
        this.render();
      });
    }

    for (const button of this.dom.presetButtons) {
      button.addEventListener("click", () => {
        this.applyPreset(button.dataset.preset);
      });
    }
  }

  applyPreset(presetName) {
    const preset = presetSettings[presetName];
    if (!preset) return;
    this.stopAutoRun();
    this.clearTransientAnimations();
    this.setActivePreset(presetName);
    const settings = Object.fromEntries(
      Object.entries(preset).filter(([key]) => key !== "label")
    );
    for (const [key, value] of Object.entries(settings)) {
      if (this.dom.sliders[key]) this.dom.sliders[key].value = value;
    }
    this.model.reset({ hiddenRoom: this.dom.hiddenRoomSelect.value, settings });
    this.model.trace.unshift(`Preset selected: ${preset.label}. The agent has been reset so the behavioural change is easier to see.`);
    this.render();
  }

  setActivePreset(presetName) {
    for (const button of this.dom.presetButtons) {
      button.classList.toggle("active", button.dataset.preset === presetName);
    }
  }

  clearTransientAnimations() {
    window.clearTimeout(this.searchTimer);
    window.clearTimeout(this.routeTimer);
    this.movementRoute = null;
    for (const roomNode of this.dom.worldMap.querySelectorAll(".room")) {
      roomNode.classList.remove("searching", "found", "missed");
    }
  }

  getSettingsFromControls() {
    return Object.fromEntries(
      Object.entries(this.dom.sliders).map(([key, slider]) => [key, Number(slider.value)])
    );
  }

  runStep({ keepAutoRunning = false } = {}) {
    if (!keepAutoRunning) this.stopAutoRun();

    const snapshot = this.model.step();
    if (snapshot.lastAction) {
      this.beginMovementAnimation(snapshot.lastAction);
    }
    this.render();
  }

  beginMovementAnimation(action) {
    this.movementRoute = {
      from: action.from,
      to: action.to,
      until: Date.now() + 820
    };

    window.clearTimeout(this.routeTimer);
    window.clearTimeout(this.searchTimer);

    this.searchTimer = window.setTimeout(() => {
      this.triggerSearchAnimation(action.searchedRoom, action.detected);
    }, 640);

    this.routeTimer = window.setTimeout(() => {
      this.movementRoute = null;
      this.render();
    }, 940);
  }

  triggerSearchAnimation(roomId, detected) {
    const roomNode = this.dom.worldMap.querySelector(`[data-room-id="${roomId}"]`);
    if (!roomNode) return;

    const burst = roomNode.querySelector(`[data-search-burst="${roomId}"]`);
    if (burst) burst.textContent = detected ? "found" : "search";

    roomNode.classList.remove("searching", "found", "missed");
    void roomNode.offsetWidth;
    roomNode.classList.add("searching", detected ? "found" : "missed");

    window.setTimeout(() => {
      roomNode.classList.remove("searching", "found", "missed");
    }, 920);
  }

  startAutoRun() {
    this.dom.autoBtn.textContent = "Pause";
    this.autoTimer = window.setInterval(() => {
      const snapshot = this.model.snapshot();
      if (snapshot.success || snapshot.observations.length >= 8) {
        this.stopAutoRun();
        return;
      }
      this.runStep({ keepAutoRunning: true });
    }, 1450);
  }

  stopAutoRun() {
    if (this.autoTimer) {
      window.clearInterval(this.autoTimer);
      this.autoTimer = null;
    }
    this.dom.autoBtn.textContent = "Auto run";
  }

  render() {
    const snapshot = this.model.snapshot();
    this.renderControlOutputs(snapshot);
    this.renderWorld(snapshot);
    this.renderBeliefs(snapshot);
    this.renderPolicies(snapshot);
    this.renderTrace(snapshot);
    this.renderAdvanced(snapshot);
  }

  renderControlOutputs(snapshot) {
    for (const [key, value] of Object.entries(snapshot.settings)) {
      const slider = this.dom.sliders[key];
      const output = this.dom.outputs[key];
      if (slider && Number(slider.value) !== Number(value)) slider.value = value;
      if (output) output.textContent = Number(value).toFixed(2);
    }
  }

  renderWorld(snapshot) {
    this.dom.agentLocationLabel.textContent = titleCaseRoom(snapshot.agentLocation);
    this.dom.hiddenRoomSelect.value = snapshot.hiddenRoom;

    for (const room of rooms) {
      const belief = snapshot.beliefs[room.id];
      const node = this.dom.worldMap.querySelector(`[data-room-id="${room.id}"]`);
      node.style.setProperty("--belief", belief.toFixed(3));
      node.classList.toggle("active", room.id === snapshot.agentLocation);
      node.classList.toggle("searched", snapshot.observations.some(obs => obs.room === room.id));
      node.classList.toggle("hidden-room", room.id === snapshot.hiddenRoom);
      node.classList.toggle("revealed", this.revealed || snapshot.success);
      const pill = node.querySelector(`[data-belief-pill="${room.id}"]`);
      if (pill) pill.textContent = formatProbability(belief);
    }

    this.renderRoutes(snapshot);

    const agent = this.dom.worldMap.querySelector("[data-agent]");
    const agentPos = roomLookup[snapshot.agentLocation].agent;
    agent.style.left = `${agentPos.x}%`;
    agent.style.top = `${agentPos.y}%`;
    agent.classList.toggle("moving", Boolean(this.movementRoute && Date.now() < this.movementRoute.until));

    const mug = this.dom.worldMap.querySelector("[data-mug-marker]");
    const mugPos = roomLookup[snapshot.hiddenRoom].mug;
    mug.style.left = `${mugPos.x}%`;
    mug.style.top = `${mugPos.y}%`;
    mug.classList.toggle("hidden", !this.revealed && !snapshot.success);
  }

  renderRoutes(snapshot) {
    const allLines = [...this.dom.worldMap.querySelectorAll(".route-line")];
    for (const line of allLines) {
      line.classList.remove("planned", "active-route");
    }

    for (const room of rooms) {
      const node = this.dom.worldMap.querySelector(`[data-room-id="${room.id}"]`);
      if (node) node.classList.remove("chosen-target");
    }

    const movementActive = this.movementRoute && Date.now() < this.movementRoute.until;
    const route = movementActive
      ? shortestPath(this.movementRoute.from, this.movementRoute.to)
      : shortestPath(snapshot.agentLocation, snapshot.policies[0]?.roomId ?? snapshot.agentLocation);
    const routeClass = movementActive ? "active-route" : "planned";

    if (route.length <= 1) {
      const target = snapshot.policies[0]?.roomId ?? snapshot.agentLocation;
      const node = this.dom.worldMap.querySelector(`[data-room-id="${target}"]`);
      if (node) node.classList.add("chosen-target");
      return;
    }

    for (let i = 0; i < route.length - 1; i += 1) {
      const key = routeKey(route[i], route[i + 1]);
      const line = this.dom.worldMap.querySelector(`[data-route-key="${key}"]`);
      if (line) line.classList.add(routeClass);
    }
  }

  renderBeliefs(snapshot) {
    this.dom.beliefBars.innerHTML = "";
    const sortedRooms = [...rooms].sort((a, b) => snapshot.beliefs[b.id] - snapshot.beliefs[a.id]);

    for (const room of sortedRooms) {
      const row = document.createElement("div");
      row.className = "belief-row";

      const name = document.createElement("div");
      name.className = "belief-name";
      name.textContent = room.name;

      const track = document.createElement("div");
      track.className = "belief-track";
      const fill = document.createElement("div");
      fill.className = "belief-fill";
      fill.style.width = `${snapshot.beliefs[room.id] * 100}%`;
      track.appendChild(fill);

      const value = document.createElement("div");
      value.className = "belief-value";
      value.textContent = formatProbability(snapshot.beliefs[room.id]);

      row.append(name, track, value);
      this.dom.beliefBars.appendChild(row);
    }

    this.dom.uncertaintyLabel.textContent = Number(snapshot.uncertainty).toFixed(2);
    this.dom.mostLikelyLabel.textContent = titleCaseRoom(snapshot.mostLikely);
    this.dom.searchCountLabel.textContent = String(snapshot.observations.length);
  }

  renderPolicies(snapshot) {
    this.dom.policyTable.innerHTML = "";

    for (const [index, policy] of snapshot.policies.entries()) {
      const row = document.createElement("div");
      row.className = `policy-row ${index === 0 ? "best" : ""}`;
      row.innerHTML = `
        <div>
          <div class="policy-title">${policy.title}</div>
          <div class="policy-small">P(target) ${formatProbability(policy.pTarget)} · distance ${policy.distance}</div>
        </div>
        <div class="policy-small">Info<br><strong>${Number(policy.infoGain).toFixed(2)}</strong></div>
        <div class="policy-small">Cost<br><strong>${Number(policy.cost).toFixed(2)}</strong></div>
        <div class="policy-small">G<br><strong>${formatSigned(policy.expectedFreeEnergy)}</strong></div>
      `;
      this.dom.policyTable.appendChild(row);
    }
  }

  renderTrace(snapshot) {
    this.dom.traceList.innerHTML = "";
    for (const item of snapshot.trace.slice(0, 8)) {
      const li = document.createElement("li");
      li.textContent = item;
      this.dom.traceList.appendChild(li);
    }
  }

  renderAdvanced(snapshot) {
    if (!this.dom.advancedToggle.checked) return;

    this.dom.priorReadout.textContent = `Semantic prior P(location | mug)\n${JSON.stringify(formatDistribution(snapshot.prior), null, 2)}\n\nPosterior q(location)\n${JSON.stringify(formatDistribution(snapshot.beliefs), null, 2)}`;

    const latestObservation = snapshot.observations.at(-1);
    this.dom.likelihoodReadout.textContent = latestObservation
      ? `Latest observation\n${JSON.stringify({ searched: titleCaseRoom(latestObservation.room), detected: latestObservation.detected }, null, 2)}\n\nLikelihood model\nP(no mug seen | mug in searched room) = 1 - sensory precision\nP(no mug seen | mug elsewhere) ≈ 0.985`
      : `No observations yet.\n\nLikelihood model\nDetection probability is controlled by sensory precision.\nFailed searches reduce belief in the searched room.`;

    this.dom.efeReadout.textContent = `Policy scores\n${JSON.stringify(snapshot.policies.map(policy => ({
      policy: policy.title,
      G: Number(policy.expectedFreeEnergy.toFixed(3)),
      goal: Number(policy.goalValue.toFixed(3)),
      epistemic: Number(policy.epistemicValue.toFixed(3)),
      cost: Number(policy.cost.toFixed(3))
    })), null, 2)}\n\nG ≈ cost - goal value - epistemic value`;
  }
}

function formatDistribution(distribution) {
  return Object.fromEntries(
    Object.entries(distribution).map(([key, value]) => [titleCaseRoom(key), Number(value.toFixed(3))])
  );
}

window.addEventListener("DOMContentLoaded", () => {
  const model = new ActiveInferenceWorldModel();
  new LabUI(model);
});
