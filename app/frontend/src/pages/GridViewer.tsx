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
  Network,
  RefreshCw,
  Maximize,
  Minimize,
  LocateFixed,
} from 'lucide-react';
import api from '../services/api';
import { useGridStore, type LiveMetrics } from '../stores/gridStore';
import type { Topology, TopologyNode, TopologyEdge } from '../types';
import { getGridSvgIcon, type GridSvgProps, TransformerSvg } from '../components/grid/GridSvgIcons';
import TransformerNode from '../components/grid/TransformerNode';
import { calculateRadialLayout } from '../components/grid/RadialLayout';

// ─── Feeder IDs for reverse flow detection ─────────────────────
const FEEDER_IDS = ['F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12'] as const;

// ─── PV capacity map (same as Dashboard) ───────────────────────
const PV_CAPACITY_KW: Record<string, number> = {
  pv_f06_factory: 5000, pv_f06_smallind: 3500, pv_f06_residential: 1640,
  pv_f07_village: 4000, pv_f07_agricultural: 3500, pv_f07_rural: 1670,
  pv_f08_commercial: 2500, pv_f08_residential: 2000, pv_f08_mixed: 940,
  pv_f09_town: 200, pv_f09_village: 150,
  pv_f10_town: 3500, pv_f10_fishing: 2000, pv_f10_coastal: 1500,
  pv_f11_hospital: 1500, pv_f11_commercial: 3500, pv_f11_apartments: 3000, pv_f11_mixedres: 2220,
  pv_f12_res1: 3000, pv_f12_res2: 2000,
};
const TOTAL_PV_CAPACITY_KW = 47320;

// ─── Custom SVG Node (same as Dashboard) ────────────────────────
function SvgGridNode({ data }: { data: any }) {
  const SvgIcon: React.FC<GridSvgProps> = data.svgIcon;
  const statusColors = {
    normal: { text: 'text-green-400' },
    low: { text: 'text-amber-400' },
    high: { text: 'text-red-400' },
  };
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
  liveMetrics?: LiveMetrics | null,
  isPlaybackActive?: boolean,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  const transformerEdges: TopologyEdge[] = [];
  const lineEdges: TopologyEdge[] = [];
  for (const edge of topology.edges) {
    if (edge.type === 'transformer') {
      transformerEdges.push(edge);
    } else {
      lineEdges.push(edge);
    }
  }

  // Build augmented topology with transformer virtual nodes
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

  const solarActive = liveMetrics ? liveMetrics.total_solar_kw > 0 : false;

  // Bus nodes
  for (const node of topology.nodes) {
    const pos = posMap.get(node.id) ?? { x: 0, y: 0 };
    const busData = gridState?.buses?.[node.id];

    const liveV = liveMetrics?.bus_voltages?.[node.id];
    const voltagePu = liveV ?? busData?.voltage_pu?.[0];
    const voltageStatus: 'normal' | 'low' | 'high' = voltagePu
      ? voltagePu < 0.95 ? 'low' : voltagePu > 1.05 ? 'high' : 'normal'
      : 'normal';

    const shortLabel = node.label.length > 14 ? node.label.substring(0, 14) + '...' : node.label;
    const nameLower = node.label.toLowerCase();
    const isSolar = nameLower.includes('pv') || nameLower.includes('solar');

    let liveOutputKw: number | null = null;
    if (isPlaybackActive && liveMetrics && isSolar) {
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
        voltageStatus,
        voltagePu,
        isSolar,
        solarOff: isSolar && isPlaybackActive && !solarActive,
        isPlaybackActive: !!isPlaybackActive,
        liveOutputKw,
      },
    });
  }

  // Transformer intermediate nodes
  for (const te of transformerEdges) {
    const virtualId = `__xfmr_${te.id}`;
    const pos = posMap.get(virtualId) ?? { x: 0, y: 0 };
    const xfmrData = gridState?.transformers?.[te.label];

    const shortLabel = te.label.length > 12 ? te.label.substring(0, 12) + '...' : te.label;

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

  // Line edges with reverse flow detection
  for (const edge of lineEdges) {
    const lineData = gridState?.lines?.[edge.label];
    const isActive = lineData?.enabled !== false;

    let isReverseFlow = false;
    if (isPlaybackActive && liveMetrics && isActive) {
      const edgeLabelLower = edge.label.toLowerCase();
      for (const fid of FEEDER_IDS) {
        if (edgeLabelLower.includes(fid.toLowerCase())) {
          const feederKey = `power_${fid}_kw` as keyof LiveMetrics;
          const feederPower = liveMetrics[feederKey] as number | undefined;
          if (feederPower != null && feederPower < 0) {
            isReverseFlow = true;
          }
          break;
        }
      }
    }

    const strokeColor = !isActive ? '#6b7280' : isReverseFlow ? '#f97316' : '#22c55e';
    const strokeWidth = isReverseFlow ? 3 : 2;

    edges.push({
      id: edge.id,
      source: isReverseFlow ? edge.target : edge.source,
      target: isReverseFlow ? edge.source : edge.target,
      type: 'smoothstep',
      animated: isActive,
      style: { stroke: strokeColor, strokeWidth },
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
          <div className="w-6 h-0.5 mr-2" style={{ backgroundColor: '#f97316' }} />
          <span className="text-slate-300">Reverse Flow (export)</span>
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

// ─── Main GridViewer Component ──────────────────────────────────
export default function GridViewer() {
  const {
    gridState,
    topology,
    setTopology,
    playbackPlaying,
    playbackStepIndex,
    liveMetrics,
  } = useGridStore();

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [rfInstance, setRfInstance] = useState<ReactFlowInstance | null>(null);
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

  useEffect(() => {
    if (gridState && !topology) {
      loadTopology();
    }
  }, [gridState, topology, loadTopology]);

  // Update ReactFlow every 15-minute step
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

  const resetView = useCallback(() => {
    rfInstance?.fitView({ padding: 0.2 });
  }, [rfInstance]);

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <Network className="w-6 h-6 mr-2 text-blue-400" />
            Grid Topology Viewer
          </h1>
          <p className="text-slate-400 text-sm">Interactive power system network visualization</p>
        </div>
      </div>

      {/* Main Content */}
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
