import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  Handle,
  Position,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Activity,
  Zap,
  AlertTriangle,
  CheckCircle,
  Network,
  RefreshCw,
  Maximize,
  Minimize,
  LocateFixed,
  Shield,
  RotateCcw,
  ChevronRight,
} from 'lucide-react';
import { useGridStore, type LiveMetrics, type FLISRPhase } from '../stores/gridStore';
import api from '../services/api';
import type { Topology, TopologyNode, TopologyEdge, FaultPayload, IsolationResult, RestorationResult } from '../types';
import { getGridSvgIcon, type GridSvgProps } from '../components/grid/GridSvgIcons';
import { TransformerSvg } from '../components/grid/GridSvgIcons';
import TransformerNode from '../components/grid/TransformerNode';
import { calculateRadialLayout } from '../components/grid/RadialLayout';

// Valid phases per fault type (mirrors backend FAULT_TYPE_PHASES)
const FAULT_TYPE_PHASES: Record<string, string[]> = {
  LG: ['A', 'B', 'C'],
  LL: ['AB', 'BC', 'CA'],
  LLG: ['ABG', 'BCG', 'CAG'],
  LLL: ['ABC'],
};

// FLISR fault types for self-healing (maps to backend FaultTypeEnum)
const FLISR_FAULT_TYPES = [
  { value: '3phase', label: '3-Phase' },
  { value: 'lg', label: 'Line-Ground' },
  { value: 'll', label: 'Line-Line' },
  { value: 'llg', label: 'Line-Line-Ground' },
];

// Restoration strategies
const RESTORATION_STRATEGIES = [
  { value: 'auto', label: 'Auto (MARL -> Heuristic)' },
  { value: 'marl', label: 'MARL + GNN' },
  { value: 'heuristic', label: 'Heuristic (Greedy)' },
];

// ─── Fault Detection Banner ───────────────────────────────────
function FaultBanner({ fault }: { fault?: FaultPayload | null }) {
  const hasActiveFault = fault?.has_active_fault === true;
  const pred = fault?.prediction;

  // State 1: No model prediction — idle (Q4: never show anything not from model)
  if (!pred && !hasActiveFault) {
    return (
      <div className="card bg-slate-800 border-slate-600 py-3 px-5">
        <div className="flex items-center space-x-3">
          <Activity className="w-6 h-6 text-slate-500" />
          <div>
            <span className="text-lg font-bold text-slate-400">IDLE</span>
            <span className="text-sm text-slate-500 ml-3">Monitoring will begin when simulation starts</span>
          </div>
        </div>
      </div>
    );
  }

  // State 2: Active fault but no prediction yet — waiting for model
  if (hasActiveFault && !pred) {
    const af = fault?.active_fault;
    return (
      <div className="card bg-amber-900/40 border-amber-600 py-3 px-5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center space-x-3">
            <Activity className="w-6 h-6 text-amber-400 animate-pulse" />
            <span className="text-lg font-bold text-amber-400">ANALYZING...</span>
          </div>
          {af && (
            <div className="flex items-center space-x-4 text-sm">
              <div>
                <span className="text-slate-400">Bus: </span>
                <span className="text-white font-mono">{af.bus}</span>
              </div>
              <div>
                <span className="text-slate-400">Type: </span>
                <span className="text-white font-mono">{af.fault_type}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // State 3: Model prediction exists — show model output
  const isFaultDetected = pred?.is_fault === true;

  // Model says normal (is_fault=false) — this IS a model output
  if (!isFaultDetected) {
    return (
      <div className="card bg-green-900/30 border-green-700 py-3 px-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <CheckCircle className="w-6 h-6 text-green-400" />
            <div>
              <span className="text-lg font-bold text-green-400">NORMAL</span>
              <span className="text-sm text-slate-400 ml-3">Model prediction: no fault detected</span>
            </div>
          </div>
          <div className="text-sm">
            <span className="text-slate-400">Confidence: </span>
            <span className="text-white font-mono font-bold">
              {((1 - (pred?.detection_confidence ?? 0)) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    );
  }

  // State 4: Model detected fault — show full prediction details
  return (
    <div className="card bg-red-900/40 border-red-600 py-3 px-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center space-x-3">
          <AlertTriangle className="w-6 h-6 text-red-400" />
          <span className="text-lg font-bold text-red-400">FAULT DETECTED</span>
        </div>
        <div className="flex items-center space-x-6 text-sm">
          <div>
            <span className="text-slate-400">Type: </span>
            <span className="text-white font-mono font-bold">{pred!.fault_type}</span>
          </div>
          <div>
            <span className="text-slate-400">Phase: </span>
            <span className="text-white font-mono font-bold">{pred!.fault_phase}</span>
          </div>
          <div>
            <span className="text-slate-400">Confidence: </span>
            <span className="text-white font-mono font-bold">
              {(pred!.detection_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-slate-400">Location: </span>
            <span className="text-white font-mono font-bold">{pred!.fault_location_bus}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Latest Prediction Cards (shown only when fault detected) ─
function LatestPredictionCards({ fault }: { fault?: FaultPayload | null }) {
  const pred = fault?.prediction;
  if (!pred || !pred.is_fault) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="card py-3 px-4">
        <p className="text-xs text-slate-400">Detection</p>
        <p className="text-xl font-bold text-red-400">FAULT</p>
        <p className="text-xs text-slate-500">{(pred.detection_confidence * 100).toFixed(1)}% confidence</p>
      </div>
      <div className="card py-3 px-4">
        <p className="text-xs text-slate-400">Fault Type</p>
        <p className="text-xl font-bold text-amber-400">{pred.fault_type}</p>
      </div>
      <div className="card py-3 px-4">
        <p className="text-xs text-slate-400">Phase</p>
        <p className="text-xl font-bold text-blue-400">{pred.fault_phase}</p>
      </div>
      <div className="card py-3 px-4">
        <p className="text-xs text-slate-400">Location</p>
        <p className="text-lg font-bold text-white truncate" title={pred.fault_location_bus}>
          {pred.fault_location_bus}
        </p>
      </div>
    </div>
  );
}

// ─── Custom SVG Node ───────────────────────────────────────────
function SvgGridNode({ data }: { data: any }) {
  const SvgIcon: React.FC<GridSvgProps> = data.svgIcon;

  type SvgStatus = 'normal' | 'low' | 'high' | 'off' | 'fault';

  // For PV/solar nodes: show 'off' status when it's nighttime
  const effectiveStatus: SvgStatus = data.isSolar && data.solarOff ? 'off' : 'normal';

  // Fault heat map: override with 'fault' status when this bus is fault-highlighted
  let displayStatus: SvgStatus = data.faultHighlight ? 'fault' : effectiveStatus;

  // FLISR visual overrides
  if (data.isFlisrFaultBus) displayStatus = 'fault';
  else if (data.isDeEnergized && !data.isRestoredNode) displayStatus = 'off';  // dark/gray for power outage
  // Restored nodes get normal status (green)

  // Container styling for FLISR states
  let containerClass = '';
  let iconWrapClass = '';
  if (data.isFlisrFaultBus) {
    containerClass = 'bg-red-900/60 rounded-lg p-1 border-2 border-red-500 animate-pulse';
    iconWrapClass = '';
  } else if (data.isDeEnergized && !data.isRestoredNode) {
    containerClass = 'bg-slate-800/80 rounded-lg p-1 border-2 border-red-800 opacity-50';
    iconWrapClass = 'grayscale';
  } else if (data.isRestoredNode) {
    containerClass = 'bg-blue-900/40 rounded-lg p-1 border-2 border-blue-500';
    iconWrapClass = '';
  }

  return (
    <div className={`relative flex flex-col items-center group transition-all duration-500`}>
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !w-2 !h-2" />
      <Handle type="target" position={Position.Left} className="!bg-slate-400 !w-2 !h-2" />

      <div className={containerClass}>
        <div className={iconWrapClass}>
          <SvgIcon size={28} status={displayStatus} />
        </div>
      </div>

      <div
        className={`font-semibold text-[10px] mt-0.5 max-w-[90px] truncate text-center ${
          data.isDeEnergized && !data.isRestoredNode ? 'text-slate-500' :
          data.isFlisrFaultBus ? 'text-red-400 font-bold' :
          data.isRestoredNode ? 'text-blue-300' :
          'text-white'
        }`}
        title={data.label}
      >
        {data.shortLabel}
      </div>

      {data.kv != null && !data.isDeEnergized && !data.isFlisrFaultBus && (
        <div className="text-[8px] text-slate-400">{data.kv} kV</div>
      )}

      {/* Show fault probability when highlighted */}
      {data.faultProb != null && !data.isFlisrFaultBus && (
        <div className="text-[9px] font-medium text-red-400">
          {(data.faultProb * 100).toFixed(1)}%
        </div>
      )}

      {/* FLISR fault label */}
      {data.isFlisrFaultBus && (
        <div className="text-[9px] font-bold text-red-400 bg-red-900/80 px-1.5 rounded mt-0.5">FAULT</div>
      )}

      {/* FLISR de-energized label */}
      {data.isDeEnergized && !data.isRestoredNode && (
        <div className="text-[8px] font-bold text-red-500 bg-red-900/60 px-1 rounded mt-0.5">NO POWER</div>
      )}

      {/* FLISR restored label */}
      {data.isRestoredNode && (
        <div className="text-[8px] font-bold text-blue-400 bg-blue-900/60 px-1 rounded mt-0.5">RESTORED</div>
      )}

      {/* Show live power output for generators during playback */}
      {data.liveOutputKw != null && data.liveOutputKw > 0 && (
        <div className="text-[8px] font-medium text-yellow-400">
          {data.liveOutputKw > 1000 ? `${(data.liveOutputKw / 1000).toFixed(1)} MW` : `${data.liveOutputKw.toFixed(0)} kW`}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-slate-400 !w-2 !h-2" />
      <Handle type="source" position={Position.Right} className="!bg-slate-400 !w-2 !h-2" />
    </div>
  );
}

const nodeTypes = {
  svgNode: SvgGridNode,
  transformerNode: TransformerNode,
};

// ─── PV capacity map for capacity-weighted output display ──────
// Keyed by substring of node label (case-insensitive match).
// Total capacity: 47320 kW across 20 PV systems.
const PV_CAPACITY_KW: Record<string, number> = {
  pv_f06_factory: 5000,
  pv_f06_smallind: 3500,
  pv_f06_residential: 1640,
  pv_f07_village: 4000,
  pv_f07_agricultural: 3500,
  pv_f07_rural: 1670,
  pv_f08_commercial: 2500,
  pv_f08_residential: 2000,
  pv_f08_mixed: 940,
  pv_f09_town: 200,
  pv_f09_village: 150,
  pv_f10_town: 3500,
  pv_f10_fishing: 2000,
  pv_f10_coastal: 1500,
  pv_f11_hospital: 1500,
  pv_f11_commercial: 3500,
  pv_f11_apartments: 3000,
  pv_f11_mixedres: 2220,
  pv_f12_res1: 3000,
  pv_f12_res2: 2000,
};
const TOTAL_PV_CAPACITY_KW = 47320;

// ─── Build ReactFlow graph with radial layout ──────────────────
function buildFlowGraph(
  topology: Topology,
  gridState: any,
  liveMetrics?: LiveMetrics | null,
  isPlaybackActive?: boolean,
  faultPayload?: FaultPayload | null,
  flisrState?: {
    phase: FLISRPhase;
    faultBus: string | null;
    deEnergizedBuses: string[];
    isolationResult: IsolationResult | null;
    restorationResult: RestorationResult | null;
    selfHealingGridState: any;
    isolationGridState: any;
  } | null,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Separate transformer edges – they become intermediate nodes
  const transformerEdges: TopologyEdge[] = [];
  const lineEdges: TopologyEdge[] = [];
  for (const edge of topology.edges) {
    if (edge.type === 'transformer') {
      transformerEdges.push(edge);
    } else {
      lineEdges.push(edge);
    }
  }

  // Build an augmented topology for layout:
  //  – add a virtual node for each transformer
  //  – replace each transformer edge with two line edges through the virtual node
  const augmentedNodes: TopologyNode[] = [...topology.nodes];
  const augmentedEdges: TopologyEdge[] = [...lineEdges];

  for (const te of transformerEdges) {
    const virtualId = `__xfmr_${te.id}`;
    augmentedNodes.push({ id: virtualId, label: te.label, type: 'transformer' });
    augmentedEdges.push({ id: `${te.id}_a`, source: te.source, target: virtualId, type: 'line', label: '' });
    augmentedEdges.push({ id: `${te.id}_b`, source: virtualId, target: te.target, type: 'line', label: '' });
  }

  const augmentedTopology: Topology = { nodes: augmentedNodes, edges: augmentedEdges };
  const posMap = calculateRadialLayout(augmentedTopology);

  // Determine solar activity from live metrics
  const solarActive = liveMetrics ? liveMetrics.total_solar_kw > 0 : false;

  // FLISR state lookups
  const flisrActive = flisrState && flisrState.phase !== 'idle';
  const flisrFaultBus = flisrState?.faultBus?.toLowerCase() ?? null;
  const flisrRestored = flisrState?.phase === 'restored';

  // Build de-energized bus set from VOLTAGE DATA (catches ALL affected buses)
  const flisrDeEnergized = new Set<string>();
  const flisrRestoredBuses = new Set<string>();

  if (flisrActive && flisrState?.selfHealingGridState) {
    const voltages = flisrState.selfHealingGridState.bus_voltages_pu ?? {};
    // After isolation: buses with near-zero voltage are de-energized
    if (!flisrRestored) {
      for (const [bus, vpu] of Object.entries(voltages)) {
        if ((vpu as number) < 0.1) {
          flisrDeEnergized.add(bus.toLowerCase());
        }
      }
    }
    // After restoration: compare CURRENT voltages with ISOLATION snapshot
    if (flisrRestored) {
      // Find buses that were de-energized during isolation (from isolation grid state snapshot)
      const isoVoltages = flisrState.isolationGridState?.bus_voltages_pu ?? {};
      const prevDeEnergized = new Set<string>();
      for (const [bus, vpu] of Object.entries(isoVoltages)) {
        if ((vpu as number) < 0.1) {
          prevDeEnergized.add(bus.toLowerCase());
        }
      }
      // Now check current voltages: still dead or restored?
      for (const [bus, vpu] of Object.entries(voltages)) {
        const busLower = bus.toLowerCase();
        if ((vpu as number) < 0.1) {
          flisrDeEnergized.add(busLower); // still de-energized after restoration
        } else if (prevDeEnergized.has(busLower)) {
          flisrRestoredBuses.add(busLower); // was de-energized, now restored!
        }
      }
    }
  } else if (flisrActive) {
    // Fallback: use isolated_zone_buses if no grid state yet
    for (const b of (flisrState?.deEnergizedBuses ?? [])) {
      flisrDeEnergized.add(b.toLowerCase());
    }
  }

  // Switch states from grid state (FIX: proper precedence with parentheses)
  const flisrSwitchStates: Record<string, boolean> | null =
    flisrState?.selfHealingGridState?.all_switch_states
    ?? (flisrState?.isolationResult
      ? (() => {
          const states: Record<string, boolean> = {};
          for (const action of (flisrState.isolationResult.switch_actions ?? [])) {
            states[action.switch_name] = action.action === 'close';
          }
          return states;
        })()
      : null);
  const flisrClosedTies = new Set(flisrState?.restorationResult?.closed_switches ?? []);

  // Create bus nodes (SVG icons)
  for (const node of topology.nodes) {
    const pos = posMap.get(node.id) ?? { x: 0, y: 0 };
    // Fault heat map: only active when is_fault=true (Q10)
    const isFaultActive = faultPayload?.prediction?.is_fault === true;
    const locProbs = faultPayload?.prediction?.location_probabilities;
    const faultProb = isFaultActive && locProbs ? (locProbs[node.id] ?? null) : null;
    let faultHighlight = faultProb !== null && faultProb > 0.01;

    // FLISR overrides
    const nodeIdLower = node.id.toLowerCase();
    const isFlisrFaultBus = flisrActive && nodeIdLower === flisrFaultBus;
    const isDeEnergized = flisrActive && flisrDeEnergized.has(nodeIdLower) && !isFlisrFaultBus;
    const isRestoredNode = flisrRestored && flisrRestoredBuses.has(nodeIdLower);

    // FLISR fault bus overrides the diagnostics fault highlight
    if (isFlisrFaultBus) faultHighlight = true;

    const shortLabel = node.label.length > 14
      ? node.label.substring(0, 14) + '...'
      : node.label;

    const nameLower = node.label.toLowerCase();
    const isSolar = nameLower.includes('pv') || nameLower.includes('solar');

    // Show live output for generation nodes during playback
    let liveOutputKw: number | null = null;
    if (isPlaybackActive && liveMetrics && isSolar) {
      // Capacity-weighted fraction: node's rated kW / total fleet kW
      const nodeCapacity = PV_CAPACITY_KW[nameLower] ?? (TOTAL_PV_CAPACITY_KW / 20);
      const fraction = nodeCapacity / TOTAL_PV_CAPACITY_KW;
      liveOutputKw = liveMetrics.total_solar_kw * fraction;
    }

    nodes.push({
      id: node.id,
      type: 'svgNode',
      position: pos,
      data: {
        label: node.label,
        shortLabel,
        kv: node.kv,
        svgIcon: getGridSvgIcon(node.label),
        faultHighlight: faultHighlight || isFlisrFaultBus,
        faultProb,
        isSolar,
        solarOff: isSolar && isPlaybackActive && !solarActive,
        isPlaybackActive: !!isPlaybackActive,
        liveOutputKw,
        // FLISR visual states
        isDeEnergized,
        isFlisrFaultBus,
        isRestoredNode,
      },
    });
  }

  // Create transformer intermediate nodes
  for (const te of transformerEdges) {
    const virtualId = `__xfmr_${te.id}`;
    const pos = posMap.get(virtualId) ?? { x: 0, y: 0 };
    const xfmrData = gridState?.transformers?.[te.label];

    const shortLabel = te.label.length > 12
      ? te.label.substring(0, 12) + '...'
      : te.label;

    nodes.push({
      id: virtualId,
      type: 'transformerNode',
      position: pos,
      data: {
        label: te.label,
        shortLabel,
        kva: xfmrData?.kva,
        loadingPercent: xfmrData?.loading_percent ?? 0,
      },
    });

    // Two edges: source → transformer → target
    // Check if either side is de-energized for FLISR coloring
    const teSrcLower = te.source.toLowerCase();
    const teTgtLower = te.target.toLowerCase();
    const teSrcDead = flisrActive && flisrDeEnergized.has(teSrcLower);
    const teTgtDead = flisrActive && flisrDeEnergized.has(teTgtLower);
    const teInDeadZone = teSrcDead || teTgtDead;

    const teColor = teInDeadZone ? '#374151' : '#22c55e';
    const teAnimated = !teInDeadZone;
    const teWidth = teInDeadZone ? 1 : 2;

    edges.push({
      id: `${te.id}_a`,
      source: te.source,
      target: virtualId,
      type: 'smoothstep',
      animated: teAnimated,
      style: { stroke: teColor, strokeWidth: teWidth },
      markerEnd: { type: MarkerType.ArrowClosed, color: teColor, width: 12, height: 12 },
    });
    edges.push({
      id: `${te.id}_b`,
      source: virtualId,
      target: te.target,
      type: 'smoothstep',
      animated: teAnimated,
      style: { stroke: teColor, strokeWidth: teWidth },
      markerEnd: { type: MarkerType.ArrowClosed, color: teColor, width: 12, height: 12 },
    });
  }

  // Create line edges
  for (const edge of lineEdges) {
    const lineData = gridState?.lines?.[edge.label];
    let isActive = lineData?.enabled !== false;
    let strokeColor = isActive ? '#22c55e' : '#6b7280';
    let animated = isActive;
    let strokeDasharray: string | undefined;
    let strokeWidth = 2;

    if (flisrActive) {
      const srcLower = edge.source.toLowerCase();
      const tgtLower = edge.target.toLowerCase();
      const srcDead = flisrDeEnergized.has(srcLower);
      const tgtDead = flisrDeEnergized.has(tgtLower);
      const srcIsFault = srcLower === flisrFaultBus;
      const tgtIsFault = tgtLower === flisrFaultBus;

      // 1) Closed tie switch = BLUE (alternative restoration path)
      if (flisrClosedTies.has(edge.label)) {
        strokeColor = '#3b82f6';
        animated = true;
        strokeDasharray = undefined;
        strokeWidth = 3;
      }
      // 2) Opened switch (CB/SEC) = DASHED RED-GRAY
      else if (flisrSwitchStates && flisrSwitchStates[edge.label] === false) {
        strokeColor = '#ef4444';
        animated = false;
        strokeDasharray = '8 4';
        strokeWidth = 2;
      }
      // 3) Edge connecting to/from fault bus = RED
      else if (srcIsFault || tgtIsFault) {
        strokeColor = '#ef4444';
        animated = false;
        strokeWidth = 2;
      }
      // 4) Edge where BOTH ends are de-energized = DARK GRAY (dead zone)
      else if (srcDead && tgtDead) {
        strokeColor = '#374151';
        animated = false;
        strokeWidth = 1;
      }
      // 5) Edge where one end is de-energized = boundary of outage
      else if (srcDead || tgtDead) {
        strokeColor = '#6b7280';
        animated = false;
        strokeDasharray = '4 4';
        strokeWidth = 2;
      }
      // 6) Edge between two restored buses = GREEN with blue tint
      else if (flisrRestored) {
        const srcRestored = flisrRestoredBuses.has(srcLower);
        const tgtRestored = flisrRestoredBuses.has(tgtLower);
        if (srcRestored || tgtRestored) {
          strokeColor = '#22c55e';
          animated = true;
          strokeWidth = 2;
        }
      }
    }

    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated,
      style: {
        stroke: strokeColor,
        strokeWidth,
        ...(strokeDasharray ? { strokeDasharray } : {}),
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: strokeColor,
        width: 12,
        height: 12,
      },
      label: edge.label,
      labelStyle: { fill: '#9ca3af', fontSize: 8, fontWeight: 500 },
      labelBgStyle: { fill: '#1e293b', fillOpacity: 0.9 },
      labelBgPadding: [3, 2] as [number, number],
      labelBgBorderRadius: 3,
    });
  }

  return { nodes, edges };
}

// ─── Fault Injection Panel ────────────────────────────────────
function FaultInjectionPanel({
  topology,
  fault,
}: {
  topology: Topology | null;
  fault?: FaultPayload | null;
}) {
  const [bus, setBus] = useState('');
  const [faultType, setFaultType] = useState('LG');
  const [phase, setPhase] = useState('A');
  const [resistance, setResistance] = useState(1);
  const [status, setStatus] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const busNames = useMemo(() => {
    if (!topology) return [];
    return topology.nodes
      .map(n => n.id)
      .filter(id => {
        const lower = id.toLowerCase();
        // Allow 33kV buses and feeder nodes (f05-f12)
        if (lower.startsWith('bus_33kv')) return true;
        if (/^f(0[5-9]|1[0-2])_node/.test(lower)) return true;
        return false;
      })
      .sort();
  }, [topology]);

  // Poll model status to know if injection is available
  useEffect(() => {
    const check = () => {
      api.getModelStatus()
        .then((s) => setModelReady(s.dss_model_loaded ?? false))
        .catch(() => setModelReady(false));
    };
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  // Reset phase when fault type changes
  useEffect(() => {
    const validPhases = FAULT_TYPE_PHASES[faultType] ?? [];
    if (!validPhases.includes(phase)) {
      setPhase(validPhases[0] ?? 'A');
    }
  }, [faultType]);

  const canInject = modelReady && !fault?.has_active_fault && bus !== '' && !submitting;
  const canClear = fault?.has_active_fault === true && !submitting;

  // After inject/clear, fetch fault status and update gridState
  const refreshFaultState = async () => {
    try {
      const resp = await api.getFaultStatus();
      // Map backend FaultStatusResponse to frontend FaultPayload shape
      const faultPayload: FaultPayload = {
        has_active_fault: resp.has_active_fault,
        active_fault: resp.active_fault,
        prediction: resp.latest_prediction as any,
        detection_latency_steps: resp.detection_latency_steps,
      };
      const store = useGridStore.getState();
      if (store.gridState) {
        store.setGridState({ ...store.gridState, fault: faultPayload });
      }
    } catch { /* ignore — WebSocket will update if sim is running */ }
  };

  const handleInject = async () => {
    setSubmitting(true);
    setStatus(null);
    try {
      const res = await api.injectFault(bus, faultType, phase, resistance);
      setStatus({ type: 'success', msg: res.message });
      // Refresh fault state (needed when no live simulation is streaming updates)
      await refreshFaultState();
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message ?? 'Injection failed' });
    } finally {
      setSubmitting(false);
    }
  };

  const handleClear = async () => {
    setSubmitting(true);
    setStatus(null);
    try {
      const res = await api.clearFault();
      setStatus({ type: 'success', msg: res.message });
      await refreshFaultState();
    } catch (err: any) {
      setStatus({ type: 'error', msg: err.message ?? 'Clear failed' });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card border-amber-700/50">
      <h3 className="font-semibold text-white mb-3 text-sm flex items-center">
        <Zap className="w-4 h-4 mr-1.5 text-amber-400" />
        Fault Injection
      </h3>

      <div className="space-y-2">
        {/* Bus selector */}
        <div>
          <label className="text-[10px] text-slate-400 block mb-0.5">Target Bus</label>
          <select
            value={bus}
            onChange={(e) => setBus(e.target.value)}
            className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
          >
            <option value="">Select bus...</option>
            {busNames.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </div>

        {/* Fault type */}
        <div>
          <label className="text-[10px] text-slate-400 block mb-0.5">Fault Type</label>
          <select
            value={faultType}
            onChange={(e) => setFaultType(e.target.value)}
            className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
          >
            {Object.keys(FAULT_TYPE_PHASES).map((ft) => (
              <option key={ft} value={ft}>{ft}</option>
            ))}
          </select>
        </div>

        {/* Phase */}
        <div>
          <label className="text-[10px] text-slate-400 block mb-0.5">Phase</label>
          <select
            value={phase}
            onChange={(e) => setPhase(e.target.value)}
            className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
          >
            {(FAULT_TYPE_PHASES[faultType] ?? []).map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        {/* Resistance */}
        <div>
          <label className="text-[10px] text-slate-400 block mb-0.5">Resistance (Ω)</label>
          <select
            value={resistance}
            onChange={(e) => setResistance(Number(e.target.value))}
            className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
          >
            {Array.from({ length: 10 }, (_, i) => i + 1).map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>

        {/* Inject / Clear buttons */}
        <div className="flex space-x-2 pt-1">
          <button
            onClick={handleInject}
            disabled={!canInject}
            className="flex-1 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-600 disabled:text-slate-400
                       text-white text-xs font-medium py-1.5 rounded transition-colors"
          >
            {submitting ? '...' : 'Inject'}
          </button>
          <button
            onClick={handleClear}
            disabled={!canClear}
            className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-slate-600 disabled:text-slate-400
                       text-white text-xs font-medium py-1.5 rounded transition-colors"
          >
            Clear
          </button>
        </div>

        {/* Status message */}
        {status && (
          <p className={`text-[10px] ${status.type === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {status.msg}
          </p>
        )}

        {/* Hint when model not ready */}
        {!modelReady && (
          <p className="text-[10px] text-slate-500">Run a simulation first to load the grid model</p>
        )}
      </div>
    </div>
  );
}

// ─── FLISR Timeline Stepper ──────────────────────────────────
function FLISRTimeline({ phase }: { phase: FLISRPhase }) {
  const steps = [
    { key: 'fault_injected', label: 'Fault Injected' },
    { key: 'isolated', label: 'Isolated' },
    { key: 'restored', label: 'Restored' },
  ];

  const phaseOrder = ['idle', 'fault_injected', 'isolating', 'isolated', 'restoring', 'restored'];
  const currentIdx = phaseOrder.indexOf(phase);

  return (
    <div className="flex items-center justify-between py-2">
      {steps.map((step, i) => {
        const stepPhaseIdx = phaseOrder.indexOf(step.key);
        const isDone = currentIdx >= stepPhaseIdx;
        const isActive = phase === step.key || (step.key === 'isolated' && phase === 'restoring');

        return (
          <div key={step.key} className="flex items-center flex-1">
            <div className="flex flex-col items-center flex-1">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 ${
                isDone
                  ? 'bg-green-600 border-green-500 text-white'
                  : isActive
                    ? 'bg-amber-600 border-amber-500 text-white animate-pulse'
                    : 'bg-slate-700 border-slate-600 text-slate-400'
              }`}>
                {isDone ? '\u2713' : i + 1}
              </div>
              <span className={`text-[9px] mt-1 ${isDone ? 'text-green-400' : 'text-slate-500'}`}>
                {step.label}
              </span>
            </div>
            {i < steps.length - 1 && (
              <ChevronRight className={`w-4 h-4 mx-1 ${isDone ? 'text-green-500' : 'text-slate-600'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Isolation Result Panel ──────────────────────────────────
function IsolationResultPanel({ result }: { result: IsolationResult }) {
  return (
    <div className="space-y-2">
      <div className="text-xs">
        <div className="flex justify-between">
          <span className="text-slate-400">Feeder:</span>
          <span className="text-white font-mono">{result.feeder}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">De-energized:</span>
          <span className="text-red-400 font-mono">{result.num_de_energized_loads} loads</span>
        </div>
      </div>
      {result.de_energized_loads.length > 0 && (
        <div className="space-y-1">
          {result.de_energized_loads.map((ld) => (
            <div key={ld} className="flex items-center text-[10px]">
              <div className="w-2 h-2 rounded-full bg-red-500 mr-1.5" />
              <span className="text-slate-300">{ld}</span>
            </div>
          ))}
        </div>
      )}
      {result.critical_loads_affected.length > 0 && (
        <div className="bg-red-900/40 border border-red-600 rounded p-2">
          <p className="text-[10px] text-red-400 font-medium">CRITICAL LOADS AFFECTED:</p>
          {result.critical_loads_affected.map((ld) => (
            <p key={ld} className="text-[10px] text-red-300">{ld}</p>
          ))}
        </div>
      )}
      <div className="space-y-1">
        <p className="text-[10px] text-slate-400 font-medium">Switch Actions:</p>
        {result.switch_actions.map((sa, i) => (
          <div key={i} className="flex items-center text-[10px]">
            <span className={`font-mono mr-1 ${sa.action === 'open' ? 'text-red-400' : 'text-green-400'}`}>
              {sa.action === 'open' ? '\u25CB' : '\u25CF'}
            </span>
            <span className="text-slate-300">{sa.switch_name} - {sa.reason}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Restoration Result Panel ────────────────────────────────
function RestorationResultPanel({ result }: { result: RestorationResult }) {
  return (
    <div className="space-y-2">
      <div className="text-xs space-y-1">
        <div className="flex justify-between">
          <span className="text-slate-400">Strategy:</span>
          <span className="text-white font-mono">{result.strategy}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Restoration:</span>
          <span className={`font-bold font-mono ${result.restoration_pct >= 100 ? 'text-green-400' : result.restoration_pct > 0 ? 'text-amber-400' : 'text-red-400'}`}>
            {result.restoration_pct.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Converged:</span>
          <span className={result.converged ? 'text-green-400' : 'text-red-400'}>
            {result.converged ? 'Yes' : 'No'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Radial:</span>
          <span className={result.radial ? 'text-green-400' : 'text-red-400'}>
            {result.radial ? 'Yes' : 'No'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Violations:</span>
          <span className="text-white">{result.voltage_violations.length}</span>
        </div>
      </div>
      {result.closed_switches.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-400 font-medium">Closed Tie Switches:</p>
          {result.closed_switches.map((sw) => (
            <div key={sw} className="flex items-center text-[10px]">
              <div className="w-2 h-2 rounded-full bg-blue-500 mr-1.5" />
              <span className="text-blue-300 font-mono">{sw}</span>
            </div>
          ))}
        </div>
      )}
      {result.de_energized_loads.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-400 font-medium">Still De-energized:</p>
          {result.de_energized_loads.map((ld) => (
            <div key={ld} className="flex items-center text-[10px]">
              <div className="w-2 h-2 rounded-full bg-red-500 mr-1.5" />
              <span className="text-red-300">{ld}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Switch State Table ────────────────────────────────────
function SwitchStateTable({ switchStates }: { switchStates: Record<string, boolean> | null }) {
  if (!switchStates) return null;

  const entries = Object.entries(switchStates).sort((a, b) => a[0].localeCompare(b[0]));

  return (
    <div className="card">
      <h3 className="font-semibold text-white mb-2 text-sm">Switch States</h3>
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {entries.map(([name, closed]) => (
          <div key={name} className="flex items-center justify-between text-[10px] py-0.5">
            <span className="text-slate-300 font-mono">{name}</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
              closed
                ? 'bg-green-900/50 text-green-400'
                : 'bg-red-900/50 text-red-400'
            }`}>
              {closed ? 'CLOSED' : 'OPEN'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── FLISR Control Panel ──────────────────────────────────
function FLISRPanel({ topology }: { topology: Topology | null }) {
  const {
    flisrPhase, faultBus, faultType, isolationResult, restorationResult,
    setFLISRPhase, setFaultInfo, setIsolationResult,
    setRestorationResult, setSelfHealingGridState, setIsolationGridState, resetFLISR,
  } = useGridStore();

  const [selectedBus, setSelectedBus] = useState('');
  const [selectedFaultType, setSelectedFaultType] = useState('3phase');
  const [selectedStrategy, setSelectedStrategy] = useState('marl');
  const [resistance] = useState(0.01);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const busNames = useMemo(() => {
    if (!topology) return [];
    return topology.nodes
      .map(n => n.id)
      .filter(id => {
        const lower = id.toLowerCase();
        if (lower.startsWith('bus_33kv')) return true;
        if (/^f(0[5-9]|1[0-2])_node/.test(lower)) return true;
        return false;
      })
      .sort();
  }, [topology]);

  const handleAutoFLISR = async () => {
    if (!selectedBus) return;
    setSubmitting(true);
    setError(null);

    try {
      // Step 1: Reset grid for clean start
      await api.resetGrid();
      setFaultInfo(selectedBus, selectedFaultType, '');
      setFLISRPhase('isolating');

      // Step 2: Isolate fault — show the OUTAGE first
      const isoResult = await api.isolateFault(selectedBus, selectedFaultType, resistance);
      setIsolationResult(isoResult);
      setFaultInfo(selectedBus, selectedFaultType, isoResult.feeder);

      // Get grid state after isolation (shows all de-energized buses)
      const isoGridState = await api.getSelfHealingGridState();
      setSelfHealingGridState(isoGridState as any);
      setIsolationGridState(isoGridState as any);  // Save for before/after comparison

      // Pause so user can SEE the power outage (3 seconds)
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Step 3: Restore service using MARL+GNN
      setFLISRPhase('restoring');
      const restResult = await api.restoreService(selectedStrategy);
      setRestorationResult(restResult);

      // Get final grid state (shows restored buses)
      const finalGridState = await api.getSelfHealingGridState();
      setSelfHealingGridState(finalGridState as any);
    } catch (err: any) {
      setError(err.message ?? 'FLISR failed');
      if (useGridStore.getState().flisrPhase === 'isolating') {
        setFLISRPhase('idle');
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleIsolate = async () => {
    if (!selectedBus) return;
    setSubmitting(true);
    setError(null);

    try {
      // Reset grid first for a clean start
      await api.resetGrid();
      setFaultInfo(selectedBus, selectedFaultType, '');
      setFLISRPhase('isolating');

      const result = await api.isolateFault(selectedBus, selectedFaultType, resistance);
      setIsolationResult(result);

      // Fetch grid state after isolation (shows all de-energized buses)
      const isoGridState = await api.getSelfHealingGridState();
      setSelfHealingGridState(isoGridState as any);
      setIsolationGridState(isoGridState as any);  // Save for before/after comparison
    } catch (err: any) {
      setError(err.message ?? 'Isolation failed');
      setFLISRPhase('fault_injected');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRestore = async () => {
    setSubmitting(true);
    setError(null);

    try {
      setFLISRPhase('restoring');
      const result = await api.restoreService(selectedStrategy);
      setRestorationResult(result);

      const gridState = await api.getSelfHealingGridState();
      setSelfHealingGridState(gridState as any);
    } catch (err: any) {
      setError(err.message ?? 'Restoration failed');
      setFLISRPhase('isolated');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.resetGrid();
      resetFLISR();
    } catch (err: any) {
      setError(err.message ?? 'Reset failed');
    } finally {
      setSubmitting(false);
    }
  };

  const isIdle = flisrPhase === 'idle';
  const isIsolated = flisrPhase === 'isolated';

  return (
    <div className="card border-blue-700/50">
      <h3 className="font-semibold text-white mb-3 text-sm flex items-center">
        <Shield className="w-4 h-4 mr-1.5 text-blue-400" />
        Self-Healing FLISR
      </h3>

      {/* Fault input (only when idle) */}
      {isIdle && (
        <div className="space-y-2">
          <div>
            <label className="text-[10px] text-slate-400 block mb-0.5">Target Bus</label>
            <select
              value={selectedBus}
              onChange={(e) => setSelectedBus(e.target.value)}
              className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
            >
              <option value="">Select bus...</option>
              {busNames.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[10px] text-slate-400 block mb-0.5">Fault Type</label>
            <select
              value={selectedFaultType}
              onChange={(e) => setSelectedFaultType(e.target.value)}
              className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
            >
              {FLISR_FAULT_TYPES.map((ft) => (
                <option key={ft.value} value={ft.value}>{ft.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[10px] text-slate-400 block mb-0.5">Strategy</label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="w-full bg-slate-700 text-white text-xs rounded px-2 py-1.5 border border-slate-600"
            >
              {RESTORATION_STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>

          <div className="flex space-x-2 pt-1">
            <button
              onClick={handleAutoFLISR}
              disabled={!selectedBus || submitting}
              className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:text-slate-400
                         text-white text-xs font-medium py-1.5 rounded transition-colors"
            >
              {submitting ? '...' : 'Auto FLISR'}
            </button>
            <button
              onClick={handleIsolate}
              disabled={!selectedBus || submitting}
              className="flex-1 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-600 disabled:text-slate-400
                         text-white text-xs font-medium py-1.5 rounded transition-colors"
            >
              Isolate
            </button>
          </div>
        </div>
      )}

      {/* Active FLISR state */}
      {!isIdle && (
        <div className="space-y-3">
          {/* Status */}
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">Bus:</span>
              <span className="text-white font-mono">{faultBus}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Type:</span>
              <span className="text-white font-mono">{faultType}</span>
            </div>
          </div>

          {/* Timeline */}
          <FLISRTimeline phase={flisrPhase} />

          {/* Isolation result */}
          {isolationResult && (
            <div>
              <p className="text-[10px] text-slate-400 font-medium mb-1 uppercase">Isolation</p>
              <IsolationResultPanel result={isolationResult} />
            </div>
          )}

          {/* Restore button (only after isolation) */}
          {isIsolated && (
            <button
              onClick={handleRestore}
              disabled={submitting}
              className="w-full bg-green-600 hover:bg-green-500 disabled:bg-slate-600
                         text-white text-xs font-medium py-1.5 rounded transition-colors"
            >
              {submitting ? 'Restoring...' : 'Restore Service'}
            </button>
          )}

          {/* Restoration result */}
          {restorationResult && (
            <div>
              <p className="text-[10px] text-slate-400 font-medium mb-1 uppercase">Restoration</p>
              <RestorationResultPanel result={restorationResult} />
            </div>
          )}

          {/* Reset button */}
          <button
            onClick={handleReset}
            disabled={submitting}
            className="w-full bg-slate-600 hover:bg-slate-500 disabled:bg-slate-700
                       text-white text-xs font-medium py-1.5 rounded transition-colors flex items-center justify-center"
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            {submitting ? '...' : 'Reset Grid'}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="text-[10px] text-red-400 mt-2">{error}</p>
      )}

      {/* Loading overlay */}
      {submitting && (
        <div className="text-center py-2">
          <div className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
}

// ─── Side Panel Components ─────────────────────────────────────

function ConnectionTypesLegend() {
  const { flisrPhase } = useGridStore();
  const showFlisr = flisrPhase !== 'idle';

  return (
    <div className="card">
      <h3 className="font-semibold text-white mb-3 text-sm">Connections</h3>
      <div className="space-y-2 text-xs">
        <div className="flex items-center">
          <div className="w-6 h-0.5 bg-green-500 mr-2" />
          <span className="text-slate-300">Active Line</span>
        </div>
        <div className="flex items-center">
          <TransformerSvg size={16} status="normal" className="mr-2" />
          <span className="text-slate-300">Transformer</span>
        </div>
        <div className="flex items-center">
          <div className="w-6 h-0.5 bg-gray-500 mr-2" />
          <span className="text-slate-300">Inactive</span>
        </div>
        {showFlisr && (
          <>
            <div className="border-t border-slate-600 pt-2 mt-2">
              <span className="text-slate-400 text-[10px] font-medium">FLISR</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse mr-2" />
              <span className="text-slate-300">Fault Bus</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-red-500/50 mr-2" />
              <span className="text-slate-300">De-energized</span>
            </div>
            <div className="flex items-center">
              <div className="w-6 h-0.5 mr-2" style={{ borderTop: '2px dashed #6b7280' }} />
              <span className="text-slate-300">Opened Switch</span>
            </div>
            <div className="flex items-center">
              <div className="w-6 h-0.5 bg-blue-500 mr-2" />
              <span className="text-slate-300">Closed Tie Switch</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full border-2 border-blue-500 mr-2" />
              <span className="text-slate-300">Restored Node</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function SelectedNodePanel({ node }: { node: any }) {
  if (!node) return null;
  return (
    <div className="card border-blue-500 border">
      <h3 className="font-semibold text-white mb-3 text-sm">Selected: {node.id}</h3>
      <div className="space-y-2 text-xs">
        {node.base_kv !== undefined && (
          <div className="flex justify-between">
            <span className="text-slate-400">Base Voltage:</span>
            <span className="text-white">{node.base_kv} kV</span>
          </div>
        )}
        {node.voltage_pu && (
          <div className="flex justify-between">
            <span className="text-slate-400">Voltage (pu):</span>
            <span className={`font-medium ${node.voltage_pu[0] < 0.95 ? 'text-amber-400'
                : node.voltage_pu[0] > 1.05 ? 'text-red-400'
                  : 'text-green-400'
              }`}>
              {node.voltage_pu[0]?.toFixed(4)}
            </span>
          </div>
        )}
        {node.load && (
          <>
            <div className="border-t border-slate-600 pt-2 mt-2">
              <span className="text-slate-300 font-medium">Connected Load:</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Power:</span>
              <span className="text-white">{node.load.kw?.toFixed(1)} kW</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Reactive:</span>
              <span className="text-white">{node.load.kvar?.toFixed(1)} kVAR</span>
            </div>
          </>
        )}
        {node.generator && (
          <>
            <div className="border-t border-slate-600 pt-2 mt-2">
              <span className="text-slate-300 font-medium">Generator:</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Type:</span>
              <span className="text-white">{node.generator.type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Output:</span>
              <span className="text-green-400">{node.generator.kw?.toFixed(1)} kW</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function NetworkStats({ topology }: { topology: Topology | null }) {
  const stats = useMemo(() => {
    if (!topology) return null;
    const genNodes = topology.nodes.filter(n =>
      n.label.toLowerCase().includes('pv') ||
      n.label.toLowerCase().includes('solar') ||
      n.label.toLowerCase().includes('wind')
    ).length;
    return {
      totalBuses: topology.nodes.length,
      totalLines: topology.edges.filter(e => e.type === 'line').length,
      transformers: topology.edges.filter(e => e.type === 'transformer').length,
      generators: genNodes,
    };
  }, [topology]);

  if (!stats) return null;

  return (
    <div className="card">
      <h3 className="font-semibold text-white mb-3 text-sm">Network Statistics</h3>
      <div className="space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-slate-400">Total Buses</span>
          <span className="text-white font-medium">{stats.totalBuses}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Total Lines</span>
          <span className="text-white font-medium">{stats.totalLines}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Transformers</span>
          <span className="text-white font-medium">{stats.transformers}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">DG Units</span>
          <span className="text-white font-medium">{stats.generators}</span>
        </div>
      </div>
    </div>
  );
}

// ─── Main Dashboard Component ──────────────────────────────────
export default function Dashboard() {
  const {
    gridState,
    lastSimDate,
    topology,
    setTopology,
    // Playback state
    playbackPlaying,
    playbackPaused,
    playbackFetching,
    playbackCurrentDate,
    playbackStepIndex,
    playbackDayIndex,
    playbackTotalDays,
    liveMetrics,
    // FLISR state
    flisrPhase,
    faultBus,
    faultType,
    faultFeeder,
    isolationResult,
    restorationResult,
    selfHealingGridState,
    isolationGridState,
    deEnergizedBuses,
    deEnergizedLoads,
  } = useGridStore();

  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const gridContainerRef = useRef<HTMLDivElement>(null);

  const isActive = playbackPlaying;

  // Load topology
  const loadTopology = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getTopology() as Topology;
      setTopology(data);
    } catch (error) {
      console.error('Failed to load topology:', error);
    } finally {
      setLoading(false);
    }
  }, [setTopology]);

  useEffect(() => {
    loadTopology();
  }, [loadTopology]);

  // Reload topology when gridState appears for the first time
  useEffect(() => {
    if (gridState && !topology) {
      loadTopology();
    }
  }, [gridState, topology, loadTopology]);

  // Build FLISR state object for topology coloring
  const flisrStateForGraph = useMemo(() => {
    if (flisrPhase === 'idle') return null;
    return {
      phase: flisrPhase,
      faultBus,
      deEnergizedBuses,
      isolationResult,
      restorationResult,
      selfHealingGridState,
      isolationGridState,
    };
  }, [flisrPhase, faultBus, deEnergizedBuses, isolationResult, restorationResult, selfHealingGridState, isolationGridState]);

  // Update ReactFlow every 15-minute step (playbackStepIndex) for live voltage/flow updates
  useEffect(() => {
    if (topology) {
      const { nodes: flowNodes, edges: flowEdges } = buildFlowGraph(
        topology,
        gridState,
        liveMetrics,
        isActive,
        gridState?.fault ?? null,
        flisrStateForGraph,
      );
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [topology, gridState, setNodes, setEdges, playbackStepIndex, isActive, flisrStateForGraph]);

  // Handle node click
  const onNodeClick = useCallback((_: any, node: Node) => {
    if (node.id.startsWith('__xfmr_')) return;

    const busData = gridState?.buses?.[node.id];
    const loadData = gridState?.loads
      ? Object.values(gridState.loads).find((l: any) => l.bus === node.id)
      : null;
    const genData = gridState?.generators
      ? Object.values(gridState.generators).find((g: any) => g.bus === node.id)
      : null;

    setSelectedNode({
      id: node.id,
      ...busData,
      load: loadData,
      generator: genData,
    });
  }, [gridState]);

  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!gridContainerRef.current) return;
    if (!isFullscreen) {
      gridContainerRef.current.requestFullscreen?.().catch(() => { });
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.().catch(() => { });
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  useEffect(() => {
    const handler = () => {
      if (!document.fullscreenElement) setIsFullscreen(false);
    };
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  // Reset view
  const resetView = useCallback(() => {
    rfInstance?.fitView({ padding: 0.2 });
  }, [rfInstance]);

  // ─── Derive display values from live metrics OR final grid state ───
  const currentHourLabel = liveMetrics
    ? `${String(Math.floor(liveMetrics.hour)).padStart(2, '0')}:${String(Math.round((liveMetrics.hour % 1) * 60)).padStart(2, '0')}`
    : null;

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Simulation Info Card - shows during playback or after a completed run */}
      {(isActive || lastSimDate) && (
        <div className="card flex items-center justify-between py-2 px-4">
          <div className="flex items-center space-x-4">
            <Zap className={`w-5 h-5 ${isActive ? 'text-green-400 animate-pulse' : 'text-blue-400'}`} />
            <div>
              <span className="text-sm text-slate-400">
                {isActive ? 'Simulating: ' : 'Last Simulation: '}
              </span>
              <span className="text-sm text-white font-mono">
                {isActive ? (playbackCurrentDate ?? '\u2014') : (lastSimDate ?? '\u2014')}
              </span>
            </div>
            {isActive && currentHourLabel && (
              <div>
                <span className="text-sm text-slate-400">Time: </span>
                <span className="text-sm text-white font-mono">{currentHourLabel}</span>
              </div>
            )}
            {isActive && playbackFetching && (
              <span className="text-xs text-blue-400 animate-pulse">Loading day data...</span>
            )}
          </div>
          {isActive && (
            <div className="flex items-center space-x-3">
              <div className="text-sm text-slate-400">
                Step {playbackStepIndex}/96
                {playbackTotalDays > 1 && (
                  <span className="ml-2 text-slate-500">
                    Day {playbackDayIndex + 1}/{playbackTotalDays}
                  </span>
                )}
              </div>
              <div className="w-32 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 transition-all duration-200"
                  style={{ width: `${(playbackStepIndex / 96) * 100}%` }}
                />
              </div>
              <span className={`text-xs ${playbackPaused ? 'text-amber-400' : 'text-green-400'}`}>
                {playbackPaused ? 'PAUSED' : 'RUNNING'}
              </span>
            </div>
          )}
        </div>
      )}

      {/* FLISR Status Banner */}
      {flisrPhase !== 'idle' && (() => {
        // Count de-energized and restored buses from voltage data
        const currentVoltages = selfHealingGridState?.bus_voltages_pu ?? {};
        const isoVoltages = isolationGridState?.bus_voltages_pu ?? {};
        const deadBusCount = Object.values(currentVoltages).filter((v: any) => v < 0.1).length;
        const isoDeadCount = Object.values(isoVoltages).filter((v: any) => v < 0.1).length;
        const restoredBusCount = flisrPhase === 'restored' ? Math.max(0, isoDeadCount - deadBusCount) : 0;

        return (
        <div className={`card py-3 px-5 ${
          flisrPhase === 'restored'
            ? 'bg-green-900/30 border-green-700'
            : flisrPhase === 'isolated' || flisrPhase === 'restoring'
              ? 'bg-amber-900/40 border-amber-600'
              : 'bg-red-900/40 border-red-600'
        }`}>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center space-x-3">
              <Shield className={`w-6 h-6 ${
                flisrPhase === 'restored' ? 'text-green-400'
                  : flisrPhase === 'isolated' ? 'text-amber-400'
                    : 'text-red-400'
              }`} />
              <span className={`text-lg font-bold ${
                flisrPhase === 'restored' ? 'text-green-400'
                  : flisrPhase === 'isolated' ? 'text-amber-400'
                    : 'text-red-400'
              }`}>
                {flisrPhase === 'restored'
                  ? `RESTORED - ${restorationResult?.restoration_pct?.toFixed(0)}% via ${restorationResult?.strategy}`
                  : flisrPhase === 'isolated'
                    ? `ISOLATED - ${faultFeeder}, ${deEnergizedLoads.length} loads de-energized`
                    : flisrPhase === 'isolating' || flisrPhase === 'restoring'
                      ? `${flisrPhase.toUpperCase()}...`
                      : `FAULT - ${faultBus}, ${faultType}`
                }
              </span>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div>
                <span className="text-slate-400">Bus: </span>
                <span className="text-white font-mono">{faultBus}</span>
              </div>
              <div>
                <span className="text-slate-400">Type: </span>
                <span className="text-white font-mono">{faultType}</span>
              </div>
              {deadBusCount > 0 && (
                <div>
                  <span className="text-slate-400">Affected buses: </span>
                  <span className="text-red-400 font-mono font-bold">{deadBusCount}</span>
                </div>
              )}
              {restoredBusCount > 0 && (
                <div>
                  <span className="text-slate-400">Buses restored: </span>
                  <span className="text-green-400 font-mono font-bold">{restoredBusCount}/{isoDeadCount}</span>
                </div>
              )}
            </div>
          </div>
        </div>
        );
      })()}

      {/* Fault Detection Banner (diagnostics - hidden during FLISR) */}
      {flisrPhase === 'idle' && <FaultBanner fault={gridState?.fault} />}

      {/* Latest Prediction Cards (fault only, hidden during FLISR) */}
      {flisrPhase === 'idle' && <LatestPredictionCards fault={gridState?.fault} />}

      {/* Main Grid Topology Viewer */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Topology View */}
        <div
          ref={gridContainerRef}
          className={`lg:col-span-3 card p-0 overflow-hidden relative ${isFullscreen ? 'fullscreen-grid' : ''
            }`}
        >
          {/* Toolbar */}
          <div className="absolute top-2 left-2 z-10 flex items-center space-x-2">
            <button
              onClick={loadTopology}
              disabled={loading}
              className="bg-slate-700/90 hover:bg-slate-600 text-white p-1.5 rounded-md text-xs flex items-center"
              title="Refresh topology"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={resetView}
              className="bg-slate-700/90 hover:bg-slate-600 text-white p-1.5 rounded-md text-xs flex items-center"
              title="Reset view"
            >
              <LocateFixed className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="bg-slate-700/90 hover:bg-slate-600 text-white p-1.5 rounded-md text-xs flex items-center"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {isFullscreen ? <Minimize className="w-3.5 h-3.5" /> : <Maximize className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* No data prompt */}
          {!topology && !loading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <Network className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400 text-sm">No topology data</p>
                <p className="text-slate-500 text-xs mt-1">
                  Run a simulation to load the grid topology
                </p>
              </div>
            </div>
          )}

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            onInit={setRfInstance}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.2}
            maxZoom={3}
            attributionPosition="bottom-right"
          >
            <Background color="#374151" gap={30} size={1} />
            <Controls className="!bg-slate-800 !border-slate-600 !rounded-lg" />
            <MiniMap
              nodeColor={(node) => {
                if (node.id.startsWith('__xfmr_')) return '#a855f7';
                if (node.data?.isFlisrFaultBus) return '#ef4444';
                if (node.data?.isDeEnergized && !node.data?.isRestoredNode) return '#6b7280';
                if (node.data?.isRestoredNode) return '#3b82f6';
                if (node.data?.faultHighlight) return '#ef4444';
                return '#22c55e';
              }}
              maskColor="rgba(0, 0, 0, 0.8)"
              style={{ background: '#1e293b', borderRadius: '8px' }}
              pannable
              zoomable
            />
          </ReactFlow>
        </div>

        {/* Side Panel */}
        <div className="space-y-3 overflow-y-auto">
          <FLISRPanel topology={topology} />
          {flisrPhase === 'idle' && (
            <FaultInjectionPanel
              topology={topology}
              fault={gridState?.fault}
            />
          )}
          <ConnectionTypesLegend />
          {flisrPhase !== 'idle' && (
            <SwitchStateTable switchStates={selfHealingGridState?.all_switch_states ?? null} />
          )}
          <SelectedNodePanel node={selectedNode} />
          <NetworkStats topology={topology} />
        </div>
      </div>
    </div>
  );
}
