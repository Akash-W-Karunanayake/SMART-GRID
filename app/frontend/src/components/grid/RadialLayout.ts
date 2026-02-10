/**
 * Radial distribution layout algorithm for power system topology.
 *
 * Root = highest-voltage source bus (33kV+).
 * BFS from root through edges to build a spanning tree.
 * Place root at center, children in concentric rings:
 *   - Ring 0: source bus (center)
 *   - Ring 1: substations / step-down transformers
 *   - Ring 2: distribution buses
 *   - Ring 3+: loads / generation
 *
 * Angular spacing proportional to subtree size.
 */

import type { Topology, TopologyNode, TopologyEdge } from '../../types';

interface TreeNode {
  id: string;
  children: TreeNode[];
  subtreeSize: number;
  depth: number;
}

/**
 * Build an adjacency list from topology edges.
 * Edges are undirected so add both directions.
 */
function buildAdjacency(edges: TopologyEdge[]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, new Set());
    if (!adj.has(e.target)) adj.set(e.target, new Set());
    adj.get(e.source)!.add(e.target);
    adj.get(e.target)!.add(e.source);
  }
  return adj;
}

/**
 * Pick root node: highest kv bus, or first bus if kv is absent.
 */
function pickRoot(nodes: TopologyNode[]): string {
  let best: TopologyNode | null = null;
  for (const n of nodes) {
    if (!best || (n.kv ?? 0) > (best.kv ?? 0)) best = n;
  }
  return best?.id ?? nodes[0]?.id ?? '';
}

/**
 * BFS to build a spanning tree from root.
 */
function buildTree(rootId: string, adj: Map<string, Set<string>>): TreeNode {
  const visited = new Set<string>();
  const root: TreeNode = { id: rootId, children: [], subtreeSize: 1, depth: 0 };
  visited.add(rootId);

  const queue: TreeNode[] = [root];
  while (queue.length > 0) {
    const current = queue.shift()!;
    const neighbors = adj.get(current.id) ?? new Set();
    for (const nid of neighbors) {
      if (visited.has(nid)) continue;
      visited.add(nid);
      const child: TreeNode = { id: nid, children: [], subtreeSize: 1, depth: current.depth + 1 };
      current.children.push(child);
      queue.push(child);
    }
  }

  // Compute subtree sizes bottom-up
  function computeSize(node: TreeNode): number {
    if (node.children.length === 0) {
      node.subtreeSize = 1;
      return 1;
    }
    node.subtreeSize = 1 + node.children.reduce((sum, c) => sum + computeSize(c), 0);
    return node.subtreeSize;
  }
  computeSize(root);

  return root;
}

export interface RadialPosition {
  id: string;
  x: number;
  y: number;
}

/**
 * Assign radial positions.
 *
 * @param tree   Spanning tree from BFS
 * @param cx     Center x
 * @param cy     Center y
 * @param ringSpacing  Pixels between concentric rings
 */
function assignPositions(
  tree: TreeNode,
  cx: number,
  cy: number,
  ringSpacing: number,
): RadialPosition[] {
  const positions: RadialPosition[] = [];

  // Root at center
  positions.push({ id: tree.id, x: cx, y: cy });

  function layout(node: TreeNode, angleStart: number, angleEnd: number) {
    const totalChildWeight = node.children.reduce((s, c) => s + c.subtreeSize, 0);
    if (totalChildWeight === 0) return;

    let currentAngle = angleStart;
    for (const child of node.children) {
      const sweep = ((child.subtreeSize / totalChildWeight) * (angleEnd - angleStart));
      const midAngle = currentAngle + sweep / 2;
      const radius = child.depth * ringSpacing;

      const x = cx + radius * Math.cos(midAngle);
      const y = cy + radius * Math.sin(midAngle);
      positions.push({ id: child.id, x, y });

      layout(child, currentAngle, currentAngle + sweep);
      currentAngle += sweep;
    }
  }

  layout(tree, 0, 2 * Math.PI);
  return positions;
}

/**
 * Main entry point: compute radial positions for all nodes.
 *
 * Any nodes not reachable from the root (disconnected) are placed
 * in a horizontal line below the main radial layout.
 */
export function calculateRadialLayout(
  topology: Topology,
  options?: {
    centerX?: number;
    centerY?: number;
    ringSpacing?: number;
  },
): Map<string, { x: number; y: number }> {
  const cx = options?.centerX ?? 600;
  const cy = options?.centerY ?? 500;
  const ringSpacing = options?.ringSpacing ?? 200;

  if (topology.nodes.length === 0) return new Map();

  const adj = buildAdjacency(topology.edges);
  const rootId = pickRoot(topology.nodes);
  const tree = buildTree(rootId, adj);
  const positions = assignPositions(tree, cx, cy, ringSpacing);

  const posMap = new Map<string, { x: number; y: number }>();
  for (const p of positions) {
    posMap.set(p.id, { x: p.x, y: p.y });
  }

  // Handle disconnected nodes
  const nodeWidth = 140;
  let disconnectedX = 50;
  const disconnectedY = cy + (tree.subtreeSize > 1 ? 3 : 1) * ringSpacing + 100;

  for (const node of topology.nodes) {
    if (!posMap.has(node.id)) {
      posMap.set(node.id, { x: disconnectedX, y: disconnectedY });
      disconnectedX += nodeWidth + 30;
    }
  }

  return posMap;
}
