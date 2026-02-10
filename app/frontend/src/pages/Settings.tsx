import { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Server, Database, Cpu, RefreshCw } from 'lucide-react';
import api from '../services/api';

export default function Settings() {
  const [health, setHealth] = useState<any>(null);
  const [circuitInfo, setCircuitInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [healthData, circuit] = await Promise.all([
        fetch('/health').then(r => r.json()),
        api.getCircuitInfo(),
      ]);
      setHealth(healthData);
      setCircuitInfo(circuit);
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const reloadModel = async () => {
    setLoading(true);
    try {
      await api.loadModel();
      await loadData();
    } catch (error) {
      console.error('Failed to reload model:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center">
          <SettingsIcon className="w-6 h-6 mr-2 text-slate-400" />
          Settings
        </h1>
        <p className="text-slate-400">System configuration and status</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Server Status */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Server className="w-5 h-5 mr-2 text-blue-400" />
            Server Status
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
              <span className="text-slate-400">API Status</span>
              <span className={`px-2 py-1 rounded text-xs ${
                health?.status === 'healthy'
                  ? 'bg-green-900 text-green-400'
                  : 'bg-red-900 text-red-400'
              }`}>
                {health?.status || 'Unknown'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
              <span className="text-slate-400">Model Loaded</span>
              <span className={`px-2 py-1 rounded text-xs ${
                health?.model_loaded
                  ? 'bg-green-900 text-green-400'
                  : 'bg-amber-900 text-amber-400'
              }`}>
                {health?.model_loaded ? 'Yes' : 'No'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
              <span className="text-slate-400">WebSocket Connections</span>
              <span className="text-white">{health?.websocket_connections || 0}</span>
            </div>
          </div>
          <button
            onClick={loadData}
            disabled={loading}
            className="btn-secondary w-full mt-4 flex items-center justify-center"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh Status
          </button>
        </div>

        {/* OpenDSS Model */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Database className="w-5 h-5 mr-2 text-green-400" />
            OpenDSS Model
          </h2>
          {circuitInfo ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Circuit Name</span>
                <span className="text-white font-mono">{circuitInfo.name}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Base Frequency</span>
                <span className="text-white">{circuitInfo.base_frequency} Hz</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Total Buses</span>
                <span className="text-white">{circuitInfo.num_buses}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Total Nodes</span>
                <span className="text-white">{circuitInfo.num_nodes}</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Total Elements</span>
                <span className="text-white">{circuitInfo.num_elements}</span>
              </div>
            </div>
          ) : (
            <p className="text-slate-400 text-center py-8">Model not loaded</p>
          )}
          <button
            onClick={reloadModel}
            disabled={loading}
            className="btn-primary w-full mt-4"
          >
            Reload Model
          </button>
        </div>

        {/* Research Components */}
        <div className="card lg:col-span-2">
          <h2 className="card-header flex items-center">
            <Cpu className="w-5 h-5 mr-2 text-purple-400" />
            Research Components Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Self-Healing</span>
                <span className="px-2 py-0.5 bg-amber-900 text-amber-400 text-xs rounded">Placeholder</span>
              </div>
              <p className="text-xs text-slate-400">MARL + GNN integration pending</p>
            </div>
            <div className="p-4 bg-slate-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Forecasting</span>
                <span className="px-2 py-0.5 bg-amber-900 text-amber-400 text-xs rounded">Placeholder</span>
              </div>
              <p className="text-xs text-slate-400">Stacked Ensemble integration pending</p>
            </div>
            <div className="p-4 bg-slate-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Diagnostics</span>
                <span className="px-2 py-0.5 bg-amber-900 text-amber-400 text-xs rounded">Placeholder</span>
              </div>
              <p className="text-xs text-slate-400">CNN-Transformer + R-GNN pending</p>
            </div>
            <div className="p-4 bg-slate-700/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white font-medium">Net Load</span>
                <span className="px-2 py-0.5 bg-amber-900 text-amber-400 text-xs rounded">Placeholder</span>
              </div>
              <p className="text-xs text-slate-400">ICEEMDAN + Transformer + GP pending</p>
            </div>
          </div>
        </div>

        {/* API Endpoints */}
        <div className="card lg:col-span-2">
          <h2 className="card-header">API Documentation</h2>
          <p className="text-slate-400 text-sm mb-4">
            Full API documentation is available at the following endpoints:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <a
              href="/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 bg-slate-700/50 rounded-lg hover:bg-slate-600/50 transition-colors"
            >
              <p className="text-white font-medium">Swagger UI</p>
              <p className="text-sm text-slate-400">/docs</p>
            </a>
            <a
              href="/redoc"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 bg-slate-700/50 rounded-lg hover:bg-slate-600/50 transition-colors"
            >
              <p className="text-white font-medium">ReDoc</p>
              <p className="text-sm text-slate-400">/redoc</p>
            </a>
          </div>
        </div>
      </div>

      {/* Project Info */}
      <div className="card bg-slate-700/50">
        <h2 className="card-header">Project Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-slate-400">Project ID</p>
            <p className="text-white">25-26J-092</p>
          </div>
          <div>
            <p className="text-slate-400">Institution</p>
            <p className="text-white">Sri Lanka Institute of Information Technology</p>
          </div>
          <div>
            <p className="text-slate-400">Supervisor</p>
            <p className="text-white">Mr. Jeewaka Perera</p>
          </div>
          <div>
            <p className="text-slate-400">Co-supervisor</p>
            <p className="text-white">Ms. Poorna Panduwawala</p>
          </div>
        </div>
      </div>
    </div>
  );
}
