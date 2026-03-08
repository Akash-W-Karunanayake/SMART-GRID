import { Shield, Activity, ArrowRight, CheckCircle, Brain, GitBranch } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function SelfHealing() {
  const navigate = useNavigate();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center">
          <Shield className="w-6 h-6 mr-2 text-blue-400" />
          Self-Healing Framework
        </h1>
        <p className="text-slate-400">
          Multi-Agent Reinforcement Learning (MARL) + Graph Neural Networks (GNN) for autonomous fault isolation and service restoration
        </p>
      </div>

      {/* Research Info Banner */}
      <div className="card bg-blue-900/30 border-blue-700">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <Activity className="w-6 h-6 text-blue-400" />
          </div>
          <div className="ml-4">
            <h3 className="text-white font-medium">Component 1: IT22053350 - Prasansa H.G.R</h3>
            <p className="text-slate-300 text-sm mt-1">
              This component implements a model-free self-healing framework combining MARL with GNNs
              to perform fast and effective autonomous service restoration in microgrids.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">MARL (DQN)</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">GNN (GCN)</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">OpenDSS</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">FLISR</span>
            </div>
          </div>
        </div>
      </div>

      {/* Go to Dashboard CTA */}
      <div className="card bg-slate-700/50 border-blue-600/50">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-medium">Interactive FLISR Demo</h3>
            <p className="text-slate-400 text-sm mt-1">
              The self-healing FLISR workflow is integrated into the Dashboard page.
              Select a fault bus, run isolation + restoration, and see the grid topology
              update in real-time.
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center"
          >
            Open Dashboard
            <ArrowRight className="w-4 h-4 ml-1.5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Architecture */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-400" />
            Architecture
          </h2>
          <div className="space-y-3 text-sm text-slate-300">
            <div>
              <p className="text-slate-400 text-xs font-medium mb-1">GNN Encoder</p>
              <p>2-layer GCN (1 &rarr; 64 &rarr; 32) produces per-node + global embeddings</p>
            </div>
            <div>
              <p className="text-slate-400 text-xs font-medium mb-1">MARL Agents</p>
              <p>5 agents (one per tie switch), each with independent DQN (Discrete(2))</p>
            </div>
            <div>
              <p className="text-slate-400 text-xs font-medium mb-1">Hybrid Q-Network</p>
              <p>flat_branch(53&rarr;64) + gnn_branch(96&rarr;64) &rarr; fusion(128&rarr;128&rarr;2)</p>
            </div>
          </div>
        </div>

        {/* Performance */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
            Benchmark Results (v3)
          </h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between text-slate-300">
              <span>MARL+GNN Restoration</span>
              <span className="text-green-400 font-mono font-bold">63.2%</span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>MARL Flat Restoration</span>
              <span className="text-amber-400 font-mono font-bold">48.8%</span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Heuristic Restoration</span>
              <span className="text-blue-400 font-mono font-bold">63.8%</span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>GNN vs Flat p-value</span>
              <span className="text-white font-mono font-bold">0.0002</span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span>Scenarios Tested</span>
              <span className="text-white font-mono">100</span>
            </div>
          </div>
        </div>

        {/* FLISR Workflow */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <GitBranch className="w-5 h-5 mr-2 text-amber-400" />
            FLISR Workflow
          </h2>
          <div className="space-y-3 text-sm">
            <div className="flex items-start">
              <div className="w-6 h-6 rounded-full bg-red-600 flex items-center justify-center text-xs text-white font-bold mr-3 flex-shrink-0">1</div>
              <div>
                <p className="text-white font-medium">Fault Detection</p>
                <p className="text-slate-400 text-xs">User specifies fault bus + type</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="w-6 h-6 rounded-full bg-amber-600 flex items-center justify-center text-xs text-white font-bold mr-3 flex-shrink-0">2</div>
              <div>
                <p className="text-white font-medium">Isolation</p>
                <p className="text-slate-400 text-xs">Rule-based CB + SEC opening</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="w-6 h-6 rounded-full bg-green-600 flex items-center justify-center text-xs text-white font-bold mr-3 flex-shrink-0">3</div>
              <div>
                <p className="text-white font-medium">Restoration</p>
                <p className="text-slate-400 text-xs">MARL+GNN selects optimal tie switches</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Grid Info */}
      <div className="card">
        <h2 className="card-header">Chunnakam GSS Grid Model</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-slate-400 text-xs">Voltage</p>
            <p className="text-white font-medium">132/33 kV</p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Feeders</p>
            <p className="text-white font-medium">8 (F05-F12)</p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Switches</p>
            <p className="text-white font-medium">21 (8 CB + 8 SEC + 5 Tie)</p>
          </div>
          <div>
            <p className="text-slate-400 text-xs">Loads</p>
            <p className="text-white font-medium">22 (1 critical)</p>
          </div>
        </div>
      </div>
    </div>
  );
}
