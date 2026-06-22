import { rooms, objectPriors, graphDistance, normalise, entropy, mostLikelyRoom, titleCaseRoom } from "./world.js";

const ROOM_IDS = rooms.map(room => room.id);
const EPS = 1e-9;

export class ActiveInferenceWorldModel {
  constructor() {
    this.defaultSettings = {
      semanticStrength: 1,
      sensoryPrecision: 0.86,
      epistemicWeight: 1.2,
      goalPreference: 2.2,
      actionCost: 0.35,
      decisionNoise: 0
    };
    this.reset();
  }

  reset(overrides = {}) {
    this.objectId = "mug";
    this.agentLocation = "living_room";
    this.hiddenRoom = overrides.hiddenRoom ?? "office";
    this.settings = { ...this.defaultSettings, ...(overrides.settings ?? {}) };
    this.observations = [];
    this.success = false;
    this.trace = [
      "The agent starts in the living room with a semantic prior: mugs are usually found in kitchens, sometimes on desks and rarely in bedrooms."
    ];
    this.recomputeBeliefs();
  }

  setHiddenRoom(roomId) {
    this.hiddenRoom = roomId;
    this.observations = [];
    this.success = false;
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
    this.uncertainty = entropy(posterior);
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
      policies: this.policies.map(policy => ({ ...policy })),
      trace: [...this.trace],
      prior: this.priorDistribution()
    };
  }
}

export function formatProbability(value) {
  return Number(value).toFixed(2);
}

export function formatSigned(value) {
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
