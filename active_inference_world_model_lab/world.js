export const rooms = [
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

export const roomLookup = Object.fromEntries(rooms.map(room => [room.id, room]));

export const graph = {
  kitchen: ["living_room", "office"],
  office: ["kitchen", "living_room", "bedroom"],
  living_room: ["kitchen", "office", "bedroom"],
  bedroom: ["living_room", "office"]
};

export const routeLines = [
  ["kitchen", "office"],
  ["kitchen", "living_room"],
  ["office", "bedroom"],
  ["living_room", "bedroom"],
  ["living_room", "office"]
];

export const objectPriors = {
  mug: {
    kitchen: 0.62,
    office: 0.24,
    living_room: 0.10,
    bedroom: 0.04
  }
};

export function titleCaseRoom(id) {
  return roomLookup[id]?.name ?? id;
}

export function graphDistance(start, goal) {
  if (start === goal) return 0;
  const visited = new Set([start]);
  const queue = [{ node: start, distance: 0 }];

  while (queue.length > 0) {
    const current = queue.shift();
    for (const next of graph[current.node] ?? []) {
      if (visited.has(next)) continue;
      if (next === goal) return current.distance + 1;
      visited.add(next);
      queue.push({ node: next, distance: current.distance + 1 });
    }
  }
  return 99;
}

export function normalise(distribution) {
  const total = Object.values(distribution).reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    const flat = 1 / rooms.length;
    return Object.fromEntries(rooms.map(room => [room.id, flat]));
  }
  return Object.fromEntries(Object.entries(distribution).map(([key, value]) => [key, value / total]));
}

export function entropy(distribution) {
  return Object.values(distribution).reduce((sum, p) => {
    if (p <= 0) return sum;
    return sum - p * Math.log2(p);
  }, 0);
}

export function mostLikelyRoom(distribution) {
  return Object.entries(distribution).sort((a, b) => b[1] - a[1])[0][0];
}
