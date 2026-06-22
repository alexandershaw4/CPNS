import { rooms, roomLookup, routeLines, titleCaseRoom } from "./world.js";
import { formatProbability, formatSigned } from "./model.js";

export class LabUI {
  constructor(model) {
    this.model = model;
    this.revealed = false;
    this.autoTimer = null;
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
      this.model.step();
      this.render();
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
      this.model.setHiddenRoom(this.dom.hiddenRoomSelect.value);
      this.render();
    });

    this.dom.advancedToggle.addEventListener("change", () => {
      this.dom.advancedPanel.classList.toggle("hidden", !this.dom.advancedToggle.checked);
      this.render();
    });

    for (const [key, slider] of Object.entries(this.dom.sliders)) {
      slider.addEventListener("input", () => {
        this.model.updateSettings({ [key]: Number(slider.value) });
        this.render();
      });
    }
  }

  getSettingsFromControls() {
    return Object.fromEntries(
      Object.entries(this.dom.sliders).map(([key, slider]) => [key, Number(slider.value)])
    );
  }

  startAutoRun() {
    this.dom.autoBtn.textContent = "Pause";
    this.autoTimer = window.setInterval(() => {
      const snapshot = this.model.snapshot();
      if (snapshot.success || snapshot.observations.length >= 8) {
        this.stopAutoRun();
        return;
      }
      this.model.step();
      this.render();
    }, 1150);
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
      const node = this.dom.worldMap.querySelector(`[data-room-id="${room.id}"]`);
      node.classList.toggle("active", room.id === snapshot.agentLocation);
      node.classList.toggle("searched", snapshot.observations.some(obs => obs.room === room.id));
      node.classList.toggle("hidden-room", room.id === snapshot.hiddenRoom);
      node.classList.toggle("revealed", this.revealed);
    }

    const agent = this.dom.worldMap.querySelector("[data-agent]");
    const agentPos = roomLookup[snapshot.agentLocation].agent;
    agent.style.left = `${agentPos.x}%`;
    agent.style.top = `${agentPos.y}%`;

    const mug = this.dom.worldMap.querySelector("[data-mug-marker]");
    const mugPos = roomLookup[snapshot.hiddenRoom].mug;
    mug.style.left = `${mugPos.x}%`;
    mug.style.top = `${mugPos.y}%`;
    mug.classList.toggle("hidden", !this.revealed && !snapshot.success);
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
