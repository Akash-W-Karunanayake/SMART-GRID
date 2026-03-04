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
} from 'lucide-react';
import { useGridStore, type LiveMetrics } from '../stores/gridStore';
import api from '../services/api';
import type { Topology, TopologyNode, TopologyEdge, FaultPayload } from '../types';
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

  // For PV/solar nodes: show 'off' status when it's nighttime
  const effectiveStatus = data.isSolar && data.solarOff ? 'off' : 'normal';

  // Fault heat map: override with 'fault' status when this bus is fault-highlighted
  const displayStatus = data.faultHighlight ? 'fault' : effectiveStatus;

  return (
    <div className={`relative flex flex-col items-center group ${data.isPlaybackActive ? 'transition-all duration-300' : ''}`}>
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !w-2 !h-2" />
      <Handle type="target" position={Position.Left} className="!bg-slate-400 !w-2 !h-2" />

      <div className={data.isPlaybackActive && !data.solarOff && data.isSolar ? 'animate-pulse' : ''}>
        <SvgIcon size={28} status={displayStatus} />
      </div>

      <div
        className="font-semibold text-white text-[10px] mt-0.5 max-w-[90px] truncate text-center"
        title={data.label}
      >
        {data.shortLabel}
      </div>

      {data.kv != null && (
        <div className="text-[8px] text-slate-400">{data.kv} kV</div>
      )}

      {/* Show fault probability when highlighted */}
      {data.faultProb != null && (
        <div className="text-[9px] font-medium text-red-400">
          {(data.faultProb * 100).toFixed(1)}%
        </div>
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

  // Create bus nodes (SVG icons)
  for (const node of topology.nodes) {
    const pos = posMap.get(node.id) ?? { x: 0, y: 0 };
    // Fault heat map: only active when is_fault=true (Q10)
    const isFaultActive = faultPayload?.prediction?.is_fault === true;
    const locProbs = faultPayload?.prediction?.location_probabilities;
    const faultProb = isFaultActive && locProbs ? (locProbs[node.id] ?? null) : null;
    const faultHighlight = faultProb !== null && faultProb > 0.01;

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
        faultHighlight,
        faultProb,
        isSolar,
        solarOff: isSolar && isPlaybackActive && !solarActive,
        isPlaybackActive: !!isPlaybackActive,
        liveOutputKw,
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

    // Two edges: source → transformer → target (green for normal flow)
    edges.push({
      id: `${te.id}_a`,
      source: te.source,
      target: virtualId,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#22c55e', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#22c55e', width: 12, height: 12 },
    });
    edges.push({
      id: `${te.id}_b`,
      source: virtualId,
      target: te.target,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#22c55e', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#22c55e', width: 12, height: 12 },
    });
  }

  // Create line edges
  for (const edge of lineEdges) {
    const lineData = gridState?.lines?.[edge.label];
    const isActive = lineData?.enabled !== false;
    const strokeColor = isActive ? '#22c55e' : '#6b7280';

    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: isActive,
      style: { stroke: strokeColor, strokeWidth: 2 },
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

// ─── Side Panel Components ─────────────────────────────────────

function ConnectionTypesLegend() {
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

  // Update ReactFlow every 15-minute step (playbackStepIndex) for live voltage/flow updates
  useEffect(() => {
    if (topology) {
      const { nodes: flowNodes, edges: flowEdges } = buildFlowGraph(
        topology,
        gridState,
        liveMetrics,
        isActive,
        gridState?.fault ?? null,
      );
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [topology, gridState, setNodes, setEdges, playbackStepIndex, isActive]);

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

      {/* Fault Detection Banner */}
      <FaultBanner fault={gridState?.fault} />

      {/* Latest Prediction Cards (fault only) */}
      <LatestPredictionCards fault={gridState?.fault} />

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
          <FaultInjectionPanel
            topology={topology}
            fault={gridState?.fault}
          />
          <ConnectionTypesLegend />
          <SelectedNodePanel node={selectedNode} />
          <NetworkStats topology={topology} />
        </div>
      </div>
    </div>
  );
}
