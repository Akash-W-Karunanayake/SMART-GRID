import { useEffect, useState, useCallback, useMemo } from 'react';
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
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Network,
  RefreshCw,
  Home,
  Building2,
  Cross,
  GraduationCap,
  Factory,
  Sun,
  Wind,
  Zap,
  CircleDot,
  Warehouse,
  Store,
  Church,
  Landmark,
  Radio,
  Droplets,
} from 'lucide-react';
import api from '../services/api';
import { useGridStore } from '../stores/gridStore';
import type { Topology, TopologyNode } from '../types';

// Icon mapping based on node name patterns
function getNodeIcon(nodeName: string): React.ElementType {
  const name = nodeName.toLowerCase();

  // Solar/PV systems
  if (name.includes('pv') || name.includes('solar')) return Sun;
  // Wind farms
  if (name.includes('wind')) return Wind;
  // Hospital
  if (name.includes('hospital') || name.includes('health')) return Cross;
  // Schools/Education
  if (name.includes('school') || name.includes('college') || name.includes('university')) return GraduationCap;
  // Industrial/Factory
  if (name.includes('factory') || name.includes('industrial') || name.includes('ind')) return Factory;
  // Commercial/Business
  if (name.includes('commercial') || name.includes('shop') || name.includes('market') || name.includes('store')) return Store;
  // Government/Public buildings
  if (name.includes('gov') || name.includes('municipal') || name.includes('office')) return Landmark;
  // Religious buildings
  if (name.includes('temple') || name.includes('church') || name.includes('mosque') || name.includes('kovil')) return Church;
  // Water pumping stations
  if (name.includes('pump') || name.includes('water')) return Droplets;
  // Telecommunication
  if (name.includes('telecom') || name.includes('tower') || name.includes('bts')) return Radio;
  // Warehouse/Storage
  if (name.includes('warehouse') || name.includes('storage')) return Warehouse;
  // Substation/Transformer nodes (high voltage)
  if (name.includes('sub') || name.includes('gss') || name.includes('33kv') || name.includes('11kv')) return Zap;
  // Residential/Houses
  if (name.includes('house') || name.includes('residential') || name.includes('home') || name.includes('lv')) return Home;
  // Buildings (general)
  if (name.includes('building') || name.includes('apartment')) return Building2;
  // Default for feeder nodes
  if (name.includes('node') || name.includes('bus')) return CircleDot;

  return CircleDot;
}

// Get node category for grouping
function getNodeCategory(nodeName: string, kv?: number): string {
  const name = nodeName.toLowerCase();

  if (kv && kv >= 33) return 'source';
  if (name.includes('pv') || name.includes('solar') || name.includes('wind')) return 'generation';
  if (name.includes('hospital') || name.includes('school') || name.includes('factory')) return 'critical';
  if (name.includes('gss') || name.includes('sub')) return 'substation';
  if (name.includes('lv') || name.includes('house')) return 'residential';
  return 'distribution';
}

// Custom node component with icons
function CustomNode({ data }: { data: any }) {
  const Icon = data.icon;
  const statusColors = {
    normal: { bg: 'bg-green-600', border: 'border-green-400', text: 'text-green-400' },
    low: { bg: 'bg-amber-600', border: 'border-amber-400', text: 'text-amber-400' },
    high: { bg: 'bg-red-600', border: 'border-red-400', text: 'text-red-400' },
  };
  const colors = statusColors[data.voltageStatus as keyof typeof statusColors] || statusColors.normal;

  return (
    <div className={`relative px-3 py-2 rounded-lg ${colors.bg} ${colors.border} border-2 shadow-lg min-w-[100px]`}>
      <Handle type="target" position={Position.Top} className="!bg-slate-400" />
      <Handle type="source" position={Position.Bottom} className="!bg-slate-400" />

      <div className="flex items-center gap-2">
        <div className="p-1.5 bg-white/20 rounded-md">
          <Icon className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-white text-xs truncate" title={data.label}>
            {data.shortLabel}
          </div>
          {data.kv && (
            <div className="text-[10px] text-white/70">{data.kv} kV</div>
          )}
        </div>
      </div>

      {data.voltagePu && (
        <div className={`mt-1 text-[10px] font-medium ${colors.text} bg-black/30 rounded px-1.5 py-0.5 text-center`}>
          {data.voltagePu.toFixed(3)} pu
        </div>
      )}
    </div>
  );
}

const nodeTypes = {
  custom: CustomNode,
};

// Improved layout algorithm with better spacing
function calculateLayout(topology: Topology, gridState: any) {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Group nodes by category
  const categories: Record<string, TopologyNode[]> = {
    source: [],
    substation: [],
    distribution: [],
    generation: [],
    critical: [],
    residential: [],
  };

  topology.nodes.forEach(node => {
    const category = getNodeCategory(node.label, node.kv);
    if (categories[category]) {
      categories[category].push(node);
    } else {
      categories.distribution.push(node);
    }
  });

  // Layout parameters
  const startY = 50;
  const levelHeight = 180;
  const nodeWidth = 140;
  const nodeSpacing = 30;

  let currentY = startY;

  // Layout each category in layers
  const categoryOrder = ['source', 'substation', 'distribution', 'generation', 'critical', 'residential'];

  categoryOrder.forEach((category) => {
    const categoryNodes = categories[category];
    if (categoryNodes.length === 0) return;

    const totalWidth = categoryNodes.length * (nodeWidth + nodeSpacing) - nodeSpacing;
    const startX = Math.max(50, (1400 - totalWidth) / 2);

    categoryNodes.forEach((node, index) => {
      const busData = gridState?.buses?.[node.id];
      const voltagePu = busData?.voltage_pu?.[0];
      const voltageStatus = voltagePu
        ? voltagePu < 0.95 ? 'low'
          : voltagePu > 1.05 ? 'high'
          : 'normal'
        : 'normal';

      // Create short label
      const shortLabel = node.label.length > 12
        ? node.label.substring(0, 12) + '...'
        : node.label;

      nodes.push({
        id: node.id,
        type: 'custom',
        position: {
          x: startX + index * (nodeWidth + nodeSpacing),
          y: currentY,
        },
        data: {
          label: node.label,
          shortLabel,
          kv: node.kv,
          icon: getNodeIcon(node.label),
          voltageStatus,
          voltagePu,
          category,
        },
      });
    });

    currentY += levelHeight;
  });

  // Create edges with improved styling
  topology.edges.forEach((edge) => {
    const lineData = gridState?.lines?.[edge.label];
    const isActive = lineData?.enabled !== false;
    const isTransformer = edge.type === 'transformer';

    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: isActive && !isTransformer,
      style: {
        stroke: isTransformer ? '#a855f7' : isActive ? '#3b82f6' : '#6b7280',
        strokeWidth: isTransformer ? 3 : 2,
        strokeDasharray: isTransformer ? '5 5' : undefined,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isTransformer ? '#a855f7' : '#3b82f6',
        width: 15,
        height: 15,
      },
      label: edge.label,
      labelStyle: {
        fill: '#9ca3af',
        fontSize: 9,
        fontWeight: 500,
      },
      labelBgStyle: {
        fill: '#1e293b',
        fillOpacity: 0.9,
      },
      labelBgPadding: [4, 2] as [number, number],
      labelBgBorderRadius: 4,
    });
  });

  return { nodes, edges };
}

export default function GridViewer() {
  const { gridState, topology, setTopology } = useGridStore();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<any>(null);

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

  // Load on mount and reload when gridState becomes available (after simulation)
  useEffect(() => {
    loadTopology();
  }, [loadTopology]);

  // Reload topology when gridState appears for the first time (pipeline just ran)
  useEffect(() => {
    if (gridState && !topology) {
      loadTopology();
    }
  }, [gridState, topology, loadTopology]);

  // Update ReactFlow when topology or grid state changes
  useEffect(() => {
    if (topology) {
      const { nodes: flowNodes, edges: flowEdges } = calculateLayout(topology, gridState);
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [topology, gridState, setNodes, setEdges]);

  // Handle node click
  const onNodeClick = useCallback((_: any, node: Node) => {
    const busData = gridState?.buses?.[node.id];
    const loadData = gridState?.loads ? Object.values(gridState.loads).find((l: any) => l.bus === node.id) : null;
    const genData = gridState?.generators ? Object.values(gridState.generators).find((g: any) => g.bus === node.id) : null;

    setSelectedNode({
      id: node.id,
      ...busData,
      load: loadData,
      generator: genData,
      category: node.data.category,
    });
  }, [gridState]);

  // Statistics
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

  return (
    <div className="space-y-6 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <Network className="w-6 h-6 mr-2 text-blue-400" />
            Grid Topology Viewer
          </h1>
          <p className="text-slate-400">Interactive power system network visualization</p>
        </div>
        <button
          onClick={loadTopology}
          disabled={loading}
          className="btn-secondary flex items-center"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-220px)]">
        {/* Topology View */}
        <div className="lg:col-span-3 card p-0 overflow-hidden">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.3}
            maxZoom={2}
            attributionPosition="bottom-right"
          >
            <Background color="#374151" gap={30} size={1} />
            <Controls className="!bg-slate-800 !border-slate-600 !rounded-lg" />
            <MiniMap
              nodeColor={(node) => {
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
        <div className="space-y-4 overflow-y-auto">
          {/* Legend */}
          <div className="card">
            <h3 className="font-semibold text-white mb-3">Voltage Status</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-green-600 border-2 border-green-400 mr-2" />
                <span className="text-slate-300">Normal (0.95-1.05 pu)</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-amber-600 border-2 border-amber-400 mr-2" />
                <span className="text-slate-300">Low Voltage (&lt;0.95 pu)</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 rounded bg-red-600 border-2 border-red-400 mr-2" />
                <span className="text-slate-300">High Voltage (&gt;1.05 pu)</span>
              </div>
            </div>
          </div>

          {/* Icon Legend */}
          <div className="card">
            <h3 className="font-semibold text-white mb-3">Component Icons</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center gap-1.5 text-slate-300">
                <Sun className="w-4 h-4 text-yellow-400" />
                <span>Solar PV</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Wind className="w-4 h-4 text-cyan-400" />
                <span>Wind Farm</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Cross className="w-4 h-4 text-red-400" />
                <span>Hospital</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <GraduationCap className="w-4 h-4 text-blue-400" />
                <span>School</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Factory className="w-4 h-4 text-orange-400" />
                <span>Industrial</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Home className="w-4 h-4 text-green-400" />
                <span>Residential</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Zap className="w-4 h-4 text-purple-400" />
                <span>Substation</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <Store className="w-4 h-4 text-pink-400" />
                <span>Commercial</span>
              </div>
            </div>
          </div>

          {/* Connection Types */}
          <div className="card">
            <h3 className="font-semibold text-white mb-3">Connections</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center">
                <div className="w-8 h-0.5 bg-blue-500 mr-2" />
                <span className="text-slate-300">Active Line</span>
              </div>
              <div className="flex items-center">
                <div className="w-8 h-0.5 bg-purple-500 mr-2" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #a855f7 0px, #a855f7 5px, transparent 5px, transparent 10px)' }} />
                <span className="text-slate-300">Transformer</span>
              </div>
              <div className="flex items-center">
                <div className="w-8 h-0.5 bg-gray-500 mr-2" />
                <span className="text-slate-300">Inactive</span>
              </div>
            </div>
          </div>

          {/* Selected Node Details */}
          {selectedNode && (
            <div className="card border-blue-500 border">
              <h3 className="font-semibold text-white mb-3">Selected: {selectedNode.id}</h3>
              <div className="space-y-2 text-sm">
                {selectedNode.base_kv !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Base Voltage:</span>
                    <span className="text-white">{selectedNode.base_kv} kV</span>
                  </div>
                )}
                {selectedNode.voltage_pu && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Voltage (pu):</span>
                    <span className={`font-medium ${
                      selectedNode.voltage_pu[0] < 0.95 ? 'text-amber-400' :
                      selectedNode.voltage_pu[0] > 1.05 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {selectedNode.voltage_pu[0]?.toFixed(4)}
                    </span>
                  </div>
                )}
                {selectedNode.load && (
                  <>
                    <div className="border-t border-slate-600 pt-2 mt-2">
                      <span className="text-slate-300 font-medium">Connected Load:</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Power:</span>
                      <span className="text-white">{selectedNode.load.kw?.toFixed(1)} kW</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Reactive:</span>
                      <span className="text-white">{selectedNode.load.kvar?.toFixed(1)} kVAR</span>
                    </div>
                  </>
                )}
                {selectedNode.generator && (
                  <>
                    <div className="border-t border-slate-600 pt-2 mt-2">
                      <span className="text-slate-300 font-medium">Generator:</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Type:</span>
                      <span className="text-white">{selectedNode.generator.type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Output:</span>
                      <span className="text-green-400">{selectedNode.generator.kw?.toFixed(1)} kW</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Network Statistics */}
          <div className="card">
            <h3 className="font-semibold text-white mb-3">Network Statistics</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Total Buses</span>
                <span className="text-white font-medium">{stats?.totalBuses || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Total Lines</span>
                <span className="text-white font-medium">{stats?.totalLines || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Transformers</span>
                <span className="text-white font-medium">{stats?.transformers || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">DG Units</span>
                <span className="text-white font-medium">{stats?.generators || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
