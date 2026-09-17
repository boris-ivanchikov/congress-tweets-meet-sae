/*! Bundles Graphology, graphology-layout-forceatlas2, and
 * graphology-communities-louvain. MIT License, copyright Guillaume Plique and
 * the Graphology contributors. */
import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import ForceAtlas2Layout from "graphology-layout-forceatlas2/worker";

const COOLING_PHASES = [
  { duration: 1200, slowDown: 5 },
  { duration: 3600, slowDown: 8 },
  { duration: 9000, slowDown: 12 },
  { duration: Infinity, slowDown: 20 },
];

function median(values) {
  if (!values.length) return 1;
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

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

class ForceAtlas2Runtime {
  constructor(nodes, edges, callbacks) {
    this.nodes = nodes;
    this.edges = edges;
    this.nodeByKey = new Map(nodes.map((node) => [String(node.id), node]));
    this.callbacks = typeof callbacks === "function" ? { onFrame: callbacks } : callbacks;
    this.graph = null;
    this.layout = null;
    this.options = null;
    this.running = false;
    this.settling = false;
    this.frame = null;
    this.phase = 0;
    this.phaseStarted = 0;
    this.settleStarted = 0;
    this.frameCount = 0;
    this.lastSample = null;
    this.movement = Infinity;
    this.stableSamples = 0;
  }

  buildGraph(options) {
    const graph = new Graph({ type: "undirected", multi: false, allowSelfLoops: false });
    for (const node of this.nodes) {
      graph.addNode(String(node.id), { x: node.x, y: node.y, size: 1 });
    }
    for (const edge of this.edges) {
      if (edge.visualWeight < options.edgeCutoff) continue;
      graph.addUndirectedEdge(String(edge.source.id), String(edge.target.id), {
        weight: Math.max(0.0001, options.attraction) * edge.visualWeight ** options.edgePower,
        visualWeight: edge.visualWeight,
        virtual: false,
      });
    }
    return graph;
  }

  createLayout(slowDown) {
    if (this.layout) this.layout.kill();
    this.layout = new ForceAtlas2Layout(this.graph, {
      getEdgeWeight: "weight",
      settings: {
        adjustSizes: false,
        barnesHutOptimize: true,
        barnesHutTheta: 0.6,
        edgeWeightInfluence: 1,
        gravity: this.options.gravity,
        linLogMode: this.options.linLog,
        outboundAttractionDistribution: false,
        scalingRatio: Math.max(0.001, this.options.repulsion),
        slowDown,
        strongGravityMode: false,
      },
    });
  }

  configure(options) {
    this.stop();
    this.options = { ...options };
    this.graph = this.buildGraph(this.options);
    this.createLayout(COOLING_PHASES[0].slowDown);
    this.settling = false;
  }

  reheat() {
    if (!this.layout) return;
    this.stop();
    this.phase = 0;
    this.phaseStarted = performance.now();
    this.settleStarted = this.phaseStarted;
    this.frameCount = 0;
    this.lastSample = null;
    this.movement = Infinity;
    this.stableSamples = 0;
    this.settling = true;
    this.createLayout(COOLING_PHASES[0].slowDown);
    this.running = true;
    this.layout.start();
    this.callbacks?.onState?.({ state: "settling", phase: this.phase });
    this.scheduleFrame();
  }

  resume() {
    if (!this.layout || this.running || !this.settling) return;
    this.running = true;
    this.layout.start();
    this.scheduleFrame();
  }

  pause() {
    this.running = false;
    if (this.layout) this.layout.stop();
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = null;
  }

  stop() {
    this.pause();
    this.settling = false;
  }

  kill() {
    this.stop();
    if (this.layout) this.layout.kill();
    this.layout = null;
    this.graph = null;
  }

  syncFromNodes() {
    if (this.options) this.configure(this.options);
  }

  advanceCooling(now) {
    const current = COOLING_PHASES[this.phase];
    if (now - this.phaseStarted < current.duration || this.phase >= COOLING_PHASES.length - 1) return;
    this.phase += 1;
    this.phaseStarted = now;
    this.layout.stop();
    if (this.phase === 2) this.recolorFromLayout({ addCohesion: true });
    this.createLayout(COOLING_PHASES[this.phase].slowDown);
    this.layout.start();
    this.callbacks?.onState?.({ state: "settling", phase: this.phase });
  }

  measureMovement() {
    const sample = [];
    let minX = Infinity; let maxX = -Infinity;
    let minY = Infinity; let maxY = -Infinity;
    for (let index = 0; index < this.nodes.length; index += 8) {
      const node = this.nodes[index];
      sample.push(node.x, node.y);
      minX = Math.min(minX, node.x); maxX = Math.max(maxX, node.x);
      minY = Math.min(minY, node.y); maxY = Math.max(maxY, node.y);
    }
    if (this.lastSample) {
      let squared = 0;
      for (let index = 0; index < sample.length; index += 1) {
        const delta = sample[index] - this.lastSample[index];
        squared += delta * delta;
      }
      const rms = Math.sqrt(squared / Math.max(1, sample.length / 2));
      const diagonal = Math.max(1e-9, Math.hypot(maxX - minX, maxY - minY));
      const normalized = rms / diagonal;
      this.movement = Number.isFinite(this.movement)
        ? this.movement * 0.72 + normalized * 0.28
        : normalized;
      this.stableSamples = this.movement < 0.000012 ? this.stableSamples + 1 : 0;
    }
    this.lastSample = sample;
  }

  addAdaptiveCohesion(groups) {
    const strength = this.options.communityCohesion;
    if (!(strength > 0)) return;
    groups.forEach((group, community) => {
      const keys = group.map((node) => String(node.id)).sort((left, right) => Number(left) - Number(right));
      if (keys.length < 4) return;
      const rng = seededRandom(73 + community);
      for (let index = 0; index < keys.length; index += 1) {
        for (let link = 0; link < 2; link += 1) {
          let other = Math.floor(rng() * keys.length);
          if (other === index) other = (other + 1) % keys.length;
          const source = keys[index];
          const target = keys[other];
          if (this.graph.hasEdge(source, target)) continue;
          this.graph.addUndirectedEdge(source, target, {
            weight: strength,
            visualWeight: 0,
            virtual: true,
          });
        }
      }
    });
  }

  recolorFromLayout({ addCohesion = false } = {}) {
    if (!this.graph || !this.graph.size) return;
    const lengths = [];
    this.graph.forEachEdge((_edge, attributes, _source, _target, sourceAttributes, targetAttributes) => {
      if (attributes.virtual) return;
      lengths.push(Math.hypot(
        targetAttributes.x - sourceAttributes.x,
        targetAttributes.y - sourceAttributes.y,
      ));
    });
    const typicalLength = Math.max(1e-9, median(lengths));
    this.graph.updateEachEdgeAttributes((_edge, attributes, _source, _target, sourceAttributes, targetAttributes) => {
      if (attributes.virtual) return attributes;
      const distance = Math.hypot(
        targetAttributes.x - sourceAttributes.x,
        targetAttributes.y - sourceAttributes.y,
      );
      const proximity = 1 / (1 + (distance / (typicalLength * 2.25)) ** 2);
      return { ...attributes, communityWeight: attributes.weight * (0.18 + 0.82 * proximity) };
    });
    const result = louvain.detailed(this.graph, {
      getEdgeWeight: (_edge, attributes) => attributes.virtual ? 0 : attributes.communityWeight,
      randomWalk: true,
      resolution: 1.35,
      rng: seededRandom(73),
    });
    const groups = new Map();
    for (const [key, community] of Object.entries(result.communities)) {
      if (!groups.has(community)) groups.set(community, []);
      groups.get(community).push(this.nodeByKey.get(key));
    }
    const ordered = [...groups.values()].sort((left, right) =>
      right.length - left.length || Math.min(...left.map((node) => node.id)) - Math.min(...right.map((node) => node.id))
    );
    ordered.forEach((group, cluster) => {
      for (const node of group) node.cluster = cluster;
    });
    if (addCohesion) this.addAdaptiveCohesion(ordered);
    this.callbacks?.onCommunities?.({ count: ordered.length, modularity: result.modularity });
  }

  finish() {
    this.recolorFromLayout();
    this.pause();
    this.settling = false;
    this.callbacks?.onState?.({
      state: "settled",
      elapsed: performance.now() - this.settleStarted,
      movement: this.movement,
    });
  }

  scheduleFrame() {
    if (!this.running || this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      this.graph.forEachNode((key, attributes) => {
        const node = this.nodeByKey.get(key);
        if (!node) return;
        node.x = attributes.x;
        node.y = attributes.y;
      });
      this.frameCount += 1;
      const now = performance.now();
      if (this.frameCount % 6 === 0) this.measureMovement();
      this.advanceCooling(now);
      this.callbacks?.onFrame?.({
        elapsed: now - this.settleStarted,
        movement: this.movement,
        phase: this.phase,
      });
      if ((this.phase === COOLING_PHASES.length - 1 && this.stableSamples >= 12) ||
          now - this.settleStarted >= 32000) {
        this.finish();
        return;
      }
      this.scheduleFrame();
    });
  }
}

window.ForceAtlas2Runtime = ForceAtlas2Runtime;
