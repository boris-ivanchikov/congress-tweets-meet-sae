#!/usr/bin/env node
/* Precompute a ForceAtlas2 layout from the browser graph payload. */

const fs = require("fs");
const Graph = require("graphology");
const forceAtlas2 = require("graphology-layout-forceatlas2");
const louvain = require("graphology-communities-louvain");

function argument(name, fallback) {
  const flag = `--${name}`;
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const input = argument("input", "viewer/data/activation_graph.json");
const output = argument("output", "/tmp/activation-forceatlas2.json");
const graphOutput = argument("graph-output", null);
const iterations = Number(argument("iterations", "500"));
const edgePower = Number(argument("edge-power", "1.6"));
const attraction = Number(argument("attraction", "1"));
const repulsion = Number(argument("repulsion", "1"));
const gravity = Number(argument("gravity", "1"));
const linLog = argument("linlog", "true") === "true";
const strongGravity = argument("strong-gravity", "false") === "true";
const outboundAttraction = argument("dissuade-hubs", "false") === "true";
const edgeWeightInfluence = Number(argument("edge-weight-influence", "1"));
const adaptiveCohesion = Number(argument("adaptive-cohesion", "0"));
const discoveryIterations = Number(argument("discovery-iterations", String(Math.floor(iterations * 0.45))));

function seededRandom(seed = 73) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function median(values) {
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function discoverCommunities(graph) {
  const lengths = [];
  graph.forEachEdge((_key, attributes, _source, _target, sourceAttributes, targetAttributes) => {
    if (attributes.virtual) return;
    lengths.push(Math.hypot(targetAttributes.x - sourceAttributes.x, targetAttributes.y - sourceAttributes.y));
  });
  const typicalLength = Math.max(1e-9, median(lengths));
  graph.updateEachEdgeAttributes((_key, attributes, _source, _target, sourceAttributes, targetAttributes) => {
    if (attributes.virtual) return attributes;
    const distance = Math.hypot(
      targetAttributes.x - sourceAttributes.x,
      targetAttributes.y - sourceAttributes.y,
    );
    const proximity = 1 / (1 + (distance / (typicalLength * 2.25)) ** 2);
    return {...attributes, communityWeight: attributes.weight * (0.18 + 0.82 * proximity)};
  });
  return louvain.detailed(graph, {
    getEdgeWeight: edge => graph.getEdgeAttribute(edge, "virtual") ? 0 : graph.getEdgeAttribute(edge, "communityWeight"),
    randomWalk: true,
    resolution: 1.35,
    rng: seededRandom(73),
  });
}

function addAdaptiveCohesion(graph, communities, strength) {
  const groups = new Map();
  for (const [key, community] of Object.entries(communities)) {
    if (!groups.has(community)) groups.set(community, []);
    groups.get(community).push(key);
  }
  for (const [community, keys] of groups) {
    if (keys.length < 4) continue;
    keys.sort((left, right) => Number(left) - Number(right));
    const rng = seededRandom(73 + Number(community));
    for (let index = 0; index < keys.length; index += 1) {
      for (let link = 0; link < 2; link += 1) {
        let other = Math.floor(rng() * keys.length);
        if (other === index) other = (other + 1) % keys.length;
        const source = keys[index];
        const target = keys[other];
        if (graph.hasEdge(source, target)) continue;
        graph.addUndirectedEdge(source, target, {
          weight: strength,
          visualWeight: 0,
          virtual: true,
        });
      }
    }
  }
}

const payload = JSON.parse(fs.readFileSync(input, "utf8"));
const graph = new Graph({type: "undirected", multi: false, allowSelfLoops: false});
for (const node of payload.nodes) {
  graph.addNode(String(node.id), {x: node.x, y: node.y, size: 1});
}

const logWeights = payload.edges
  .map(edge => Math.log(Math.max(edge.weight, 1e-7)))
  .sort((a, b) => a - b);
const quantile = fraction => logWeights[
  Math.min(logWeights.length - 1, Math.floor(logWeights.length * fraction))
];
const lowWeight = quantile(0.08);
const highWeight = quantile(0.98);
for (const edge of payload.edges) {
  const logWeight = Math.log(Math.max(edge.weight, 1e-7));
  const visualWeight = Math.max(0.03, Math.min(1, (logWeight - lowWeight) / (highWeight - lowWeight)));
  graph.addUndirectedEdge(String(edge.source), String(edge.target), {
    weight: attraction * visualWeight ** edgePower,
  });
}

const startedAt = Date.now();
function runLayout(runIterations) {
forceAtlas2.assign(graph, {
  iterations: runIterations,
  getEdgeWeight: "weight",
  settings: {
    adjustSizes: false,
    barnesHutOptimize: true,
    barnesHutTheta: 0.6,
    edgeWeightInfluence,
    gravity,
    linLogMode: linLog,
    outboundAttractionDistribution: outboundAttraction,
    scalingRatio: repulsion,
    slowDown: 1 + Math.log(Math.max(2, graph.order)),
    strongGravityMode: strongGravity,
  },
});
}

let adaptiveCommunities = null;
if (adaptiveCohesion > 0) {
  runLayout(Math.min(iterations, discoveryIterations));
  adaptiveCommunities = discoverCommunities(graph);
  addAdaptiveCohesion(graph, adaptiveCommunities.communities, adaptiveCohesion);
  runLayout(Math.max(0, iterations - discoveryIterations));
  adaptiveCommunities = discoverCommunities(graph);
} else {
  runLayout(iterations);
}

const positions = {};
graph.forEachNode((key, attributes) => {
  positions[key] = [attributes.x, attributes.y];
});
fs.writeFileSync(output, JSON.stringify({
  settings: {
    iterations, edgePower, attraction, repulsion, gravity, linLog,
    strongGravity, outboundAttraction, edgeWeightInfluence,
    adaptiveCohesion, discoveryIterations,
  },
  communities: adaptiveCommunities?.communities || null,
  modularity: adaptiveCommunities?.modularity ?? null,
  elapsed_seconds: (Date.now() - startedAt) / 1000,
  positions,
}));
if (graphOutput) {
  for (const node of payload.nodes) {
    const point = positions[String(node.id)];
    node.x = Number(point[0].toFixed(6));
    node.y = Number(point[1].toFixed(6));
    if (adaptiveCommunities) node.cluster = adaptiveCommunities.communities[String(node.id)];
  }
  payload.method.layout = adaptiveCommunities
    ? "two-stage ForceAtlas2 LinLog layout with emergent, layout-aware community cohesion"
    : "precomputed global ForceAtlas2 LinLog layout (independent of communities)";
  payload.method.forceatlas2 = {
    iterations, edge_power: edgePower, attraction, repulsion, gravity,
    lin_log: linLog, strong_gravity: strongGravity,
    dissuade_hubs: outboundAttraction, edge_weight_influence: edgeWeightInfluence,
    adaptive_cohesion: adaptiveCohesion,
    discovery_iterations: discoveryIterations,
  };
  fs.writeFileSync(graphOutput, JSON.stringify(payload));
}
console.log(JSON.stringify({output, graphOutput, elapsed_seconds: (Date.now() - startedAt) / 1000}));
