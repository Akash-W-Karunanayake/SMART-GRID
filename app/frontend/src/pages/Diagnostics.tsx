import { useState, useEffect, useCallback } from 'react';
import {
  AlertTriangle,
  Activity,
  Clock,
  Zap,
  RefreshCw,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useGridStore } from '../stores/gridStore';
import api from '../services/api';
import type { FaultPayload, SubCycleWaveform } from '../types';

// Phase colors for waveform plots
const PHASE_COLORS = { A: '#ef4444', B: '#3b82f6', C: '#22c55e' };

export default function Diagnostics() {
  const { gridState } = useGridStore();
  const fault: FaultPayload | undefined = gridState?.fault;
  const pred = fault?.prediction;

  const [history, setHistory] = useState<Array<Record<string, any>>>([]);
  const [waveform, setWaveform] = useState<SubCycleWaveform | null>(null);
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean } | null>(null);
  const [loadingWaveform, setLoadingWaveform] = useState(false);

  // Fetch history + model status on mount
  useEffect(() => {
    api.getFaultHistory().then(setHistory).catch(console.error);
    api.getModelStatus().then(setModelStatus).catch(console.error);
  }, []);

  // Refresh history when fault changes
  useEffect(() => {
    api.getFaultHistory().then(setHistory).catch(console.error);
  }, [fault?.has_active_fault]);

  // Fetch waveform data
  const fetchWaveform = useCallback(async () => {
    setLoadingWaveform(true);
    try {
      const data = await api.getFaultWaveform();
      setWaveform(data);
    } catch {
      setWaveform(null);
    } finally {
      setLoadingWaveform(false);
    }
  }, []);

  // Auto-fetch waveform when a new prediction arrives
  useEffect(() => {
    if (pred?.is_fault) {
      fetchWaveform();
    }
  }, [pred?.step_detected, fetchWaveform]);

  // Build waveform chart data for a specific bus
  const buildVoltageChartData = useCallback((busIdx: number) => {
    if (!waveform) return [];
    return Array.from({ length: waveform.total_cycles }, (_, cycle) => ({
      cycle,
      Va_mag: waveform.voltage_seq[cycle]?.[busIdx]?.[0] ?? 0,
      Vb_mag: waveform.voltage_seq[cycle]?.[busIdx]?.[2] ?? 0,
      Vc_mag: waveform.voltage_seq[cycle]?.[busIdx]?.[4] ?? 0,
    }));
  }, [waveform]);

  const buildCurrentChartData = useCallback((branchIdx: number) => {
    if (!waveform) return [];
    return Array.from({ length: waveform.total_cycles }, (_, cycle) => ({
      cycle,
      Ia_mag: waveform.current_seq[cycle]?.[branchIdx]?.[0] ?? 0,
      Ib_mag: waveform.current_seq[cycle]?.[branchIdx]?.[2] ?? 0,
      Ic_mag: waveform.current_seq[cycle]?.[branchIdx]?.[4] ?? 0,
    }));
  }, [waveform]);

  // Find the bus and branch index for the fault location
  const faultBusIdx = waveform && pred
    ? waveform.bus_names.indexOf(pred.fault_location_bus)
    : -1;
  const faultBranchIdx = 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <AlertTriangle className="w-6 h-6 mr-2 text-amber-400" />
            Fault Diagnostics
          </h1>
          <p className="text-slate-400 text-sm">
            CNN-Transformer + R-GNN hybrid model — real-time fault analysis
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            modelStatus?.loaded
              ? 'bg-green-900/50 text-green-400 border border-green-700'
              : 'bg-red-900/50 text-red-400 border border-red-700'
          }`}>
            Model: {modelStatus?.loaded ? 'Loaded' : 'Not Loaded'}
          </div>
        </div>
      </div>

      {/* Latest Prediction Detail */}
      <div className={`card ${pred?.is_fault ? 'border-red-600 bg-red-900/20' : 'border-slate-600'}`}>
        <h2 className="card-header flex items-center text-sm">
          <Zap className="w-4 h-4 mr-2 text-amber-400" />
          Latest Prediction
        </h2>
        {pred ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-xs text-slate-400">Detection</p>
              <p className={`text-xl font-bold ${pred.is_fault ? 'text-red-400' : 'text-green-400'}`}>
                {pred.is_fault ? 'FAULT' : 'NORMAL'}
              </p>
              <p className="text-xs text-slate-500">{(pred.detection_confidence * 100).toFixed(1)}% confidence</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-xs text-slate-400">Fault Type</p>
              <p className="text-xl font-bold text-amber-400">{pred.fault_type}</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-xs text-slate-400">Phase</p>
              <p className="text-xl font-bold text-blue-400">{pred.fault_phase}</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-xs text-slate-400">Location</p>
              <p className="text-lg font-bold text-white truncate" title={pred.fault_location_bus}>
                {pred.fault_location_bus}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-slate-500 text-sm mt-2">No prediction data. Inject a fault to see results.</p>
        )}
      </div>

      {/* Waveform Plots */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <h2 className="card-header flex items-center text-sm m-0">
            <Activity className="w-4 h-4 mr-2 text-blue-400" />
            Sub-Cycle Waveforms (20 cycles)
          </h2>
          <button
            onClick={fetchWaveform}
            disabled={loadingWaveform}
            className="text-xs text-slate-400 hover:text-white flex items-center"
          >
            <RefreshCw className={`w-3 h-3 mr-1 ${loadingWaveform ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {waveform && faultBusIdx >= 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Voltage waveform at fault bus */}
            <div>
              <h4 className="text-xs text-slate-400 mb-2">
                Voltage (pu) at {pred?.fault_location_bus ?? 'fault bus'}
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={buildVoltageChartData(faultBusIdx)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="cycle" tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: 'Cycle', position: 'bottom', fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="Va_mag" stroke={PHASE_COLORS.A} dot={false} name="Phase A" strokeWidth={1.5} />
                  <Line type="monotone" dataKey="Vb_mag" stroke={PHASE_COLORS.B} dot={false} name="Phase B" strokeWidth={1.5} />
                  <Line type="monotone" dataKey="Vc_mag" stroke={PHASE_COLORS.C} dot={false} name="Phase C" strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Current waveform at first branch */}
            <div>
              <h4 className="text-xs text-slate-400 mb-2">
                Current (A) — Branch: {waveform.branch_names[faultBranchIdx] ?? 'N/A'}
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={buildCurrentChartData(faultBranchIdx)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="cycle" tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: 'Cycle', position: 'bottom', fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #475569', fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="Ia_mag" stroke={PHASE_COLORS.A} dot={false} name="Phase A" strokeWidth={1.5} />
                  <Line type="monotone" dataKey="Ib_mag" stroke={PHASE_COLORS.B} dot={false} name="Phase B" strokeWidth={1.5} />
                  <Line type="monotone" dataKey="Ic_mag" stroke={PHASE_COLORS.C} dot={false} name="Phase C" strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <p className="text-slate-500 text-sm">
            {loadingWaveform ? 'Loading waveform data...' : 'No waveform data. Inject a fault to capture sub-cycle snapshots.'}
          </p>
        )}

        {/* Pre/fault/post cycle markers */}
        {waveform && (
          <div className="flex items-center justify-center space-x-4 mt-3 text-[10px]">
            <div className="flex items-center">
              <div className="w-3 h-1 bg-green-500 mr-1" />
              <span className="text-slate-400">Pre-fault (0-4)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-1 bg-red-500 mr-1" />
              <span className="text-slate-400">Fault (5-14)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-1 bg-blue-500 mr-1" />
              <span className="text-slate-400">Post-fault (15-19)</span>
            </div>
          </div>
        )}
      </div>

      {/* Fault Event History */}
      <div className="card">
        <h2 className="card-header flex items-center text-sm">
          <Clock className="w-4 h-4 mr-2 text-slate-400" />
          Fault Event History (Last 20)
        </h2>
        {history.length > 0 ? (
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 px-2">Bus</th>
                  <th className="text-left py-2 px-2">Type</th>
                  <th className="text-left py-2 px-2">Phase</th>
                  <th className="text-right py-2 px-2">R (Ohm)</th>
                  <th className="text-right py-2 px-2">Step Injected</th>
                  <th className="text-right py-2 px-2">Injected At</th>
                  <th className="text-right py-2 px-2">Cleared At</th>
                </tr>
              </thead>
              <tbody>
                {history.slice().reverse().map((evt, i) => (
                  <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-1.5 px-2 text-white font-mono">{evt.bus}</td>
                    <td className="py-1.5 px-2 text-amber-400">{evt.fault_type}</td>
                    <td className="py-1.5 px-2 text-blue-400">{evt.phase}</td>
                    <td className="py-1.5 px-2 text-right text-slate-300">{evt.resistance}</td>
                    <td className="py-1.5 px-2 text-right text-slate-300">{evt.step_injected}</td>
                    <td className="py-1.5 px-2 text-right text-slate-400">
                      {evt.injected_at ? new Date(evt.injected_at * 1000).toLocaleTimeString() : '\u2014'}
                    </td>
                    <td className="py-1.5 px-2 text-right text-slate-400">
                      {evt.cleared_at ? new Date(evt.cleared_at * 1000).toLocaleTimeString() : '\u2014'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-slate-500 text-sm mt-2">No fault events recorded yet.</p>
        )}
      </div>

      {/* Fault Types Reference */}
      <div className="card bg-slate-700/30">
        <h2 className="card-header text-sm">Fault Types Reference</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-2">
          {[
            { type: 'LG', desc: 'Line-to-Ground' },
            { type: 'LL', desc: 'Line-to-Line' },
            { type: 'LLG', desc: 'Line-to-Line-to-Ground' },
            { type: 'LLL', desc: 'Three-Phase' },
            { type: 'HIF', desc: 'High-Impedance' },
          ].map((f) => (
            <div key={f.type} className="flex items-center p-2 bg-slate-800 rounded text-xs">
              <span className="font-mono text-amber-400 font-bold mr-2">{f.type}</span>
              <span className="text-slate-400">{f.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
