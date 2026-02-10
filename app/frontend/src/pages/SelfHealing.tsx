import { useState, useEffect } from 'react';
import { Shield, Play, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import api from '../services/api';

export default function SelfHealing() {
  const [status, setStatus] = useState<any>(null);
  const [agentStatus, setAgentStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [triggerBus, setTriggerBus] = useState('');
  const [triggerResult, setTriggerResult] = useState<any>(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const [healingStatus, agents] = await Promise.all([
        api.getSelfHealingStatus(),
        api.getAgentStatus(),
      ]);
      setStatus(healingStatus);
      setAgentStatus(agents);
    } catch (error) {
      console.error('Failed to load status:', error);
    }
  };

  const triggerSelfHealing = async () => {
    if (!triggerBus) return;
    setLoading(true);
    try {
      const result = await api.triggerSelfHealing(triggerBus, 'LG');
      setTriggerResult(result);
    } catch (error) {
      console.error('Failed to trigger self-healing:', error);
    } finally {
      setLoading(false);
    }
  };

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
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">MARL</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">GNN</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">OpenAI Gym</span>
              <span className="px-2 py-1 bg-blue-800 text-blue-200 text-xs rounded">DQN/DDPG</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
            System Health
          </h2>
          <div className="flex items-center justify-center py-8">
            <div className="relative">
              <svg className="w-32 h-32">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  fill="none"
                  stroke="#334155"
                  strokeWidth="8"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="8"
                  strokeDasharray={`${(status?.system_health || 100) * 3.52} 352`}
                  strokeLinecap="round"
                  transform="rotate(-90 64 64)"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-3xl font-bold text-white">
                  {status?.system_health?.toFixed(0) || 100}%
                </span>
              </div>
            </div>
          </div>
          <div className="text-center text-slate-400 text-sm">
            Overall system health based on grid state analysis
          </div>
        </div>

        {/* MARL Agents */}
        <div className="card">
          <h2 className="card-header">MARL Agents Status</h2>
          <div className="space-y-3">
            {agentStatus?.agents?.map((agent: any) => (
              <div key={agent.agent_id} className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <div>
                  <p className="text-white font-medium">{agent.agent_id}</p>
                  <p className="text-sm text-slate-400">{agent.type} - {agent.target}</p>
                </div>
                <span className={`px-2 py-1 text-xs rounded ${
                  agent.status === 'active' ? 'bg-green-900 text-green-400' : 'bg-slate-600 text-slate-400'
                }`}>
                  {agent.status}
                </span>
              </div>
            ))}
            {!agentStatus?.agents?.length && (
              <p className="text-slate-400 text-center py-4">No agents configured</p>
            )}
          </div>
          <div className="mt-4 text-sm text-slate-400">
            <span className="font-medium">Mode:</span> {agentStatus?.coordination_mode || 'Decentralized'}
          </div>
        </div>

        {/* Fault Trigger (for testing) */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-amber-400" />
            Test Self-Healing
          </h2>
          <p className="text-slate-400 text-sm mb-4">
            Trigger a fault at a bus to test the self-healing response
          </p>
          <div className="flex space-x-2">
            <input
              type="text"
              value={triggerBus}
              onChange={(e) => setTriggerBus(e.target.value)}
              placeholder="Enter bus name..."
              className="input-field flex-1"
            />
            <button
              onClick={triggerSelfHealing}
              disabled={loading || !triggerBus}
              className="btn-primary flex items-center"
            >
              <Play className="w-4 h-4 mr-1" />
              Trigger
            </button>
          </div>
          {triggerResult && (
            <div className="mt-4 p-3 bg-slate-700 rounded-lg">
              <p className="text-sm text-green-400 mb-2">Restoration plan generated:</p>
              <div className="space-y-1">
                {triggerResult.restoration_plan?.map((action: any, i: number) => (
                  <p key={i} className="text-xs text-slate-300">
                    {i + 1}. {action.action_type} → {action.target_element}
                  </p>
                ))}
              </div>
              <p className="text-xs text-slate-400 mt-2">
                Estimated restoration: {triggerResult.estimated_restoration_time_seconds}s |
                Load restored: {triggerResult.load_restored_percent}%
              </p>
            </div>
          )}
        </div>

        {/* Active Faults */}
        <div className="card">
          <h2 className="card-header">Active Faults</h2>
          {status?.active_faults?.length > 0 ? (
            <div className="space-y-2">
              {status.active_faults.map((fault: any) => (
                <div key={fault.fault_id} className="p-3 bg-red-900/30 border border-red-700 rounded-lg">
                  <p className="text-red-400 font-medium">{fault.fault_type} at {fault.location}</p>
                  <p className="text-sm text-slate-400">ID: {fault.fault_id}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-2" />
              <p className="text-slate-400">No active faults</p>
            </div>
          )}
        </div>
      </div>

      {/* Integration Note */}
      <div className="card bg-slate-700/50">
        <p className="text-slate-400 text-sm">
          <span className="text-amber-400 font-medium">Note:</span> This interface shows placeholder data.
          The actual MARL+GNN models will be integrated when the ML components are complete.
          The framework is designed for seamless integration with PyTorch/TensorFlow models.
        </p>
      </div>
    </div>
  );
}
