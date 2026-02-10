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
  Sun,
  AlertTriangle,
  Thermometer,
  CheckCircle,
  Network,
  RefreshCw,
  Maximize,
  Minimize,
  LocateFixed,
} from 'lucide-react';
import { useGridStore } from '../stores/gridStore';
import api from '../services/api';
import type { Topology, TopologyNode, TopologyEdge } from '../types';
import { getGridSvgIcon, type GridSvgProps } from '../components/grid/GridSvgIcons';
import { TransformerSvg } from '../components/grid/GridSvgIcons';
import TransformerNode from '../components/grid/TransformerNode';
import { calculateRadialLayout } from '../components/grid/RadialLayout';

// ─── Stat Card ─────────────────────────────────────────────────
interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ElementType;
  color?: string;
}

function StatCard({ title, value, unit, icon: Icon, color = 'blue' }: StatCardProps) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-900/50 border-blue-700 text-blue-400',
    green: 'bg-green-900/50 border-green-700 text-green-400',
    amber: 'bg-amber-900/50 border-amber-700 text-amber-400',
    red: 'bg-red-900/50 border-red-700 text-red-400',
  };

  return (
    <div className={`card ${colorClasses[color] ?? colorClasses.blue}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-400">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">
            {value}
            {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
          </p>
        </div>
        <Icon className="w-8 h-8 opacity-50" />
      </div>
    </div>
  );
}

// ─── Custom SVG Node ───────────────────────────────────────────
function SvgGridNode({ data }: { data: any }) {
  const SvgIcon: React.FC<GridSvgProps> = data.svgIcon;
  const statusColors = {
    normal: { text: 'text-green-400' },
    low: { text: 'text-amber-400' },
    high: { text: 'text-red-400' },
  };
  // For PV/solar nodes: show 'off' status when it's nighttime
  const effectiveStatus = data.isSolar && data.solarOff ? 'off' : (data.voltageStatus ?? 'normal');
  const colors = statusColors[effectiveStatus as keyof typeof statusColors] || statusColors.normal;

  return (
    <div className={`relative flex flex-col items-center group ${data.isPlaybackActive ? 'transition-all duration-300' : ''}`}>
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !w-2 !h-2" />
      <Handle type="target" position={Position.Left} className="!bg-slate-400 !w-2 !h-2" />

      <div className={data.isPlaybackActive && !data.solarOff && data.isSolar ? 'animate-pulse' : ''}>
        <SvgIcon size={28} status={effectiveStatus} />
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

      {data.voltagePu != null && (
        <div className={`text-[9px] font-medium ${colors.text}`}>
          {data.voltagePu.toFixed(3)} pu
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

// ─── Build ReactFlow graph with radial layout ──────────────────
function buildFlowGraph(
  topology: Topology,
  gridState: any,
  liveMetrics?: { hour: number; total_solar_kw: number; total_generation_kw: number } | null,
  isPlaybackActive?: boolean,
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

  // Determine day/night and solar activity from live metrics
  const currentHour = liveMetrics?.hour ?? null;
  const isDaytime = currentHour !== null && currentHour >= 6 && currentHour < 18;
  const solarActive = liveMetrics ? liveMetrics.total_solar_kw > 0 : false;

  // Create bus nodes (SVG icons)
  for (const node of topology.nodes) {
    const pos = posMap.get(node.id) ?? { x: 0, y: 0 };
    const busData = gridState?.buses?.[node.id];
    const voltagePu = busData?.voltage_pu?.[0];
    const voltageStatus: 'normal' | 'low' | 'high' = voltagePu
      ? voltagePu < 0.95 ? 'low'
        : voltagePu > 1.05 ? 'high'
        : 'normal'
      : 'normal';

    const shortLabel = node.label.length > 14
      ? node.label.substring(0, 14) + '...'
      : node.label;

    const nameLower = node.label.toLowerCase();
    const isSolar = nameLower.includes('pv') || nameLower.includes('solar');
    const isWind = nameLower.includes('wind');
    const isGenerator = nameLower.includes('gen') || nameLower.includes('ujps');

    // Show live output for generation nodes during playback
    let liveOutputKw: number | null = null;
    if (isPlaybackActive && liveMetrics) {
      if (isSolar) liveOutputKw = liveMetrics.total_solar_kw / 20; // approx per-unit (20 PV systems)
      else if (isWind) liveOutputKw = null; // wind data available but no per-unit split needed
      else if (isGenerator) liveOutputKw = null;
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
        voltageStatus,
        voltagePu,
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

  // Create line edges (green for active, red for overloaded, gray for inactive)
  for (const edge of lineEdges) {
    const lineData = gridState?.lines?.[edge.label];
    const isActive = lineData?.enabled !== false;
    const isOverloaded = false; // TODO: detect overload from lineData if available

    const strokeColor = !isActive ? '#6b7280' : isOverloaded ? '#ef4444' : '#22c55e';

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

// ─── Side Panel Components ─────────────────────────────────────
function VoltageStatusLegend() {
  return (
    <div className="card">
      <h3 className="font-semibold text-white mb-3 text-sm">Voltage Status</h3>
      <div className="space-y-2 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-green-500 mr-2" />
          <span className="text-slate-300">Normal (0.95-1.05 pu)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-amber-500 mr-2" />
          <span className="text-slate-300">Low (&lt;0.95 pu)</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-2" />
          <span className="text-slate-300">High (&gt;1.05 pu)</span>
        </div>
      </div>
    </div>
  );
}

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
            <span className={`font-medium ${
              node.voltage_pu[0] < 0.95 ? 'text-amber-400'
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
    pipelineSteps,
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
    playbackVisibleSteps,
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

  // Compute a coarse "hour bucket" that changes every hour (not every 15 min)
  // to avoid excessive graph rebuilds during playback
  const hourBucket = liveMetrics ? Math.floor(liveMetrics.hour) : -1;
  const solarIsActive = liveMetrics ? liveMetrics.total_solar_kw > 0 : false;

  // Update ReactFlow when topology, grid state, or playback hour changes
  useEffect(() => {
    if (topology) {
      const { nodes: flowNodes, edges: flowEdges } = buildFlowGraph(
        topology,
        gridState,
        liveMetrics,
        isActive,
      );
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [topology, gridState, setNodes, setEdges, hourBucket, solarIsActive, isActive]);

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
      gridContainerRef.current.requestFullscreen?.().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
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

  // ─── Derive stat card values from live metrics OR final grid state ───
  const currentHourLabel = liveMetrics
    ? `${String(Math.floor(liveMetrics.hour)).padStart(2, '0')}:${String(Math.round((liveMetrics.hour % 1) * 60)).padStart(2, '0')}`
    : null;

  // During playback: use liveMetrics; after playback: use gridState
  const displayLoad = isActive && liveMetrics
    ? (liveMetrics.total_power_kw / 1000).toFixed(2) + ' MW'
    : gridState?.summary.total_load_kw != null
      ? gridState.summary.total_load_kw.toFixed(1)
      : '\u2014';
  const displayLoadUnit = isActive && liveMetrics ? '' : 'kW';

  const displaySolar = isActive && liveMetrics
    ? liveMetrics.total_solar_kw.toFixed(1)
    : gridState?.summary.total_solar_kw?.toFixed(1) ?? '\u2014';

  const displayGeneration = isActive && liveMetrics
    ? (liveMetrics.total_generation_kw / 1000).toFixed(2) + ' MW'
    : gridState?.summary.total_generation_kw != null
      ? (gridState.summary.total_generation_kw / 1000).toFixed(2) + ' MW'
      : '\u2014';
  const displayGenerationUnit = '';

  const displayLosses = isActive && liveMetrics
    ? liveMetrics.total_losses_kw.toFixed(1)
    : gridState?.summary.total_losses_kw?.toFixed(2) ?? '\u2014';

  const visibleSteps = isActive ? playbackVisibleSteps : pipelineSteps;
  const convergedCount = visibleSteps.filter(s => s.converged).length;
  const convergedLabel = visibleSteps.length > 0
    ? `${convergedCount}/${visibleSteps.length}`
    : '\u2014';

  const displayViolations = isActive && liveMetrics
    ? liveMetrics.voltage_violations
    : gridState?.summary.num_voltage_violations ?? 0;

  const convergedColor = isActive && liveMetrics
    ? (liveMetrics.converged ? 'green' : 'amber')
    : (gridState?.converged ? 'green' : 'amber');

  const violationColor = displayViolations > 0 ? 'red' : 'green';

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

      {/* Stats Grid - 6 cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard
          title="Total Load"
          value={displayLoad}
          unit={displayLoadUnit}
          icon={Activity}
          color="blue"
        />
        <StatCard
          title="Total Generation"
          value={displayGeneration}
          unit={displayGenerationUnit}
          icon={Zap}
          color="green"
        />
        <StatCard
          title="Solar Generation"
          value={displaySolar}
          unit="kW"
          icon={Sun}
          color={isActive && liveMetrics && liveMetrics.total_solar_kw > 0 ? 'green' : 'amber'}
        />
        <StatCard
          title="System Losses"
          value={displayLosses}
          unit="kW"
          icon={Thermometer}
          color="amber"
        />
        <StatCard
          title="Convergence"
          value={convergedLabel}
          icon={CheckCircle}
          color={convergedColor}
        />
        <StatCard
          title="Violations"
          value={displayViolations}
          icon={AlertTriangle}
          color={violationColor}
        />
      </div>

      {/* Main Grid Topology Viewer */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Topology View */}
        <div
          ref={gridContainerRef}
          className={`lg:col-span-3 card p-0 overflow-hidden relative ${
            isFullscreen ? 'fullscreen-grid' : ''
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
                const status = node.data?.voltageStatus;
                if (status === 'low') return '#f59e0b';
                if (status === 'high') return '#ef4444';
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
          <VoltageStatusLegend />
          <ConnectionTypesLegend />
          <SelectedNodePanel node={selectedNode} />
          <NetworkStats topology={topology} />
        </div>
      </div>
    </div>
  );
}
