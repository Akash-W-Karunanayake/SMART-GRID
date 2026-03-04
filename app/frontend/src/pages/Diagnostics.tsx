import { useState, useEffect } from 'react';
import {
  AlertTriangle,
  Clock,
} from 'lucide-react';
import { useGridStore } from '../stores/gridStore';
import api from '../services/api';
import type { FaultPayload } from '../types';

export default function Diagnostics() {
  const { gridState } = useGridStore();
  const fault: FaultPayload | undefined = gridState?.fault;

  const [history, setHistory] = useState<Array<Record<string, any>>>([]);
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean } | null>(null);

  // Fetch history + model status on mount
  useEffect(() => {
    api.getFaultHistory().then(setHistory).catch(console.error);
    api.getModelStatus().then(setModelStatus).catch(console.error);
  }, []);

  // Refresh history when fault changes
  useEffect(() => {
    api.getFaultHistory().then(setHistory).catch(console.error);
  }, [fault?.has_active_fault]);

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
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${modelStatus?.loaded
            ? 'bg-green-900/50 text-green-400 border border-green-700'
            : 'bg-red-900/50 text-red-400 border border-red-700'
            }`}>
            Model: {modelStatus?.loaded ? 'Loaded' : 'Not Loaded'}
          </div>
        </div>
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
                  <th className="text-left py-2 px-2">Detected Type</th>
                  <th className="text-left py-2 px-2">Detected Phase</th>
                  <th className="text-left py-2 px-2">Detected Location</th>
                  <th className="text-right py-2 px-2">Confidence</th>
                  <th className="text-right py-2 px-2">Detected At</th>
                  <th className="text-right py-2 px-2">Cleared At</th>
                </tr>
              </thead>
              <tbody>
                {history.slice().reverse().map((evt, i) => (
                  <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-1.5 px-2 text-amber-400">{evt.detected_type ?? '—'}</td>
                    <td className="py-1.5 px-2 text-blue-400">{evt.detected_phase ?? '—'}</td>
                    <td className="py-1.5 px-2 text-white font-mono">{evt.detected_location ?? '—'}</td>
                    <td className="py-1.5 px-2 text-right text-slate-300">
                      {evt.confidence != null ? `${(evt.confidence * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right text-slate-400">
                      {evt.detected_at ? new Date(evt.detected_at * 1000).toLocaleTimeString() : '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right text-slate-400">
                      {evt.cleared_at ? new Date(evt.cleared_at * 1000).toLocaleTimeString() : '—'}
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
