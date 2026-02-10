import { useState, useEffect } from 'react';
import { AlertTriangle, Search, Zap, Activity, CheckCircle } from 'lucide-react';
import api from '../services/api';
import type { DiagnosticResult } from '../types';

export default function Diagnostics() {
  const [diagnosticResult, setDiagnosticResult] = useState<DiagnosticResult | null>(null);
  const [hifResult, setHifResult] = useState<any>(null);
  const [faultHistory, setFaultHistory] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [history] = await Promise.all([
        api.getFaultHistory(),
      ]);
      setFaultHistory(history);
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const runDiagnostics = async () => {
    setLoading(true);
    try {
      const result = await api.detectFault();
      setDiagnosticResult(result);
    } catch (error) {
      console.error('Diagnostics failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const runHIFDetection = async () => {
    setLoading(true);
    try {
      const result = await api.detectHIF();
      setHifResult(result);
    } catch (error) {
      console.error('HIF detection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center">
          <AlertTriangle className="w-6 h-6 mr-2 text-amber-400" />
          Fault Diagnostics
        </h1>
        <p className="text-slate-400">
          Multi-Modal Spatio-Temporal Graph Transformer for enhanced power system fault detection
        </p>
      </div>

      {/* Research Info Banner */}
      <div className="card bg-amber-900/30 border-amber-700">
        <div className="flex items-start">
          <Zap className="w-6 h-6 text-amber-400 flex-shrink-0" />
          <div className="ml-4">
            <h3 className="text-white font-medium">Component 3: IT22577924 - Karunanayake K.P.A.W</h3>
            <p className="text-slate-300 text-sm mt-1">
              Hybrid deep learning framework combining CNN-based Transformer for temporal sequence learning
              with Recurrent Graph Neural Network (R-GNN) for spatial analysis. Capable of detecting faults,
              identifying their type and phase, and estimating their location.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-amber-800 text-amber-200 text-xs rounded">CNN-Transformer</span>
              <span className="px-2 py-1 bg-amber-800 text-amber-200 text-xs rounded">R-GNN</span>
              <span className="px-2 py-1 bg-amber-800 text-amber-200 text-xs rounded">PyTorch Geometric</span>
              <span className="px-2 py-1 bg-amber-800 text-amber-200 text-xs rounded">HIF Detection</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Run Diagnostics */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Search className="w-5 h-5 mr-2 text-blue-400" />
            Fault Detection
          </h2>
          <p className="text-slate-400 text-sm mb-4">
            Run the CNN-Transformer + R-GNN model to detect faults in real-time data
          </p>
          <button
            onClick={runDiagnostics}
            disabled={loading}
            className="btn-primary w-full mb-4"
          >
            {loading ? 'Analyzing...' : 'Run Fault Detection'}
          </button>

          {diagnosticResult && (
            <div className={`p-4 rounded-lg ${
              diagnosticResult.fault_detected
                ? 'bg-red-900/30 border border-red-700'
                : 'bg-green-900/30 border border-green-700'
            }`}>
              <div className="flex items-center mb-2">
                {diagnosticResult.fault_detected ? (
                  <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-green-400 mr-2" />
                )}
                <span className={`font-semibold ${
                  diagnosticResult.fault_detected ? 'text-red-400' : 'text-green-400'
                }`}>
                  {diagnosticResult.fault_detected ? 'Fault Detected' : 'No Fault Detected'}
                </span>
              </div>

              {diagnosticResult.fault_detected && (
                <div className="space-y-1 text-sm">
                  <p className="text-slate-300">
                    <span className="text-slate-400">Type:</span> {diagnosticResult.fault_type}
                  </p>
                  <p className="text-slate-300">
                    <span className="text-slate-400">Phase:</span> {diagnosticResult.fault_phase}
                  </p>
                  <p className="text-slate-300">
                    <span className="text-slate-400">Location:</span> {diagnosticResult.fault_location}
                  </p>
                </div>
              )}

              <p className="text-xs text-slate-400 mt-2">
                Confidence: {(diagnosticResult.confidence * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>

        {/* HIF Detection */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Activity className="w-5 h-5 mr-2 text-purple-400" />
            High-Impedance Fault (HIF) Detection
          </h2>
          <p className="text-slate-400 text-sm mb-4">
            Specialized detection for HIFs using Emanuel arc model signatures
          </p>
          <button
            onClick={runHIFDetection}
            disabled={loading}
            className="btn-secondary w-full mb-4"
          >
            {loading ? 'Analyzing...' : 'Detect HIFs'}
          </button>

          {hifResult && (
            <div className={`p-4 rounded-lg ${
              hifResult.hif_detected
                ? 'bg-purple-900/30 border border-purple-700'
                : 'bg-green-900/30 border border-green-700'
            }`}>
              <div className="flex items-center mb-2">
                {hifResult.hif_detected ? (
                  <AlertTriangle className="w-5 h-5 text-purple-400 mr-2" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-green-400 mr-2" />
                )}
                <span className={`font-semibold ${
                  hifResult.hif_detected ? 'text-purple-400' : 'text-green-400'
                }`}>
                  {hifResult.hif_detected ? 'HIF Detected' : 'No HIF Detected'}
                </span>
              </div>

              {hifResult.hif_detected && (
                <div className="space-y-1 text-sm">
                  <p className="text-slate-300">
                    <span className="text-slate-400">Location:</span> {hifResult.location}
                  </p>
                  <p className="text-slate-300">
                    <span className="text-slate-400">Fault Current:</span> {hifResult.fault_current_amps} A
                  </p>
                  {hifResult.characteristics && (
                    <>
                      <p className="text-slate-300">
                        <span className="text-slate-400">Arc Detected:</span> {hifResult.characteristics.arc_detected ? 'Yes' : 'No'}
                      </p>
                      <p className="text-slate-300">
                        <span className="text-slate-400">Harmonic Distortion:</span> {(hifResult.characteristics.harmonic_distortion * 100).toFixed(1)}%
                      </p>
                    </>
                  )}
                </div>
              )}

              <p className="text-xs text-slate-400 mt-2">
                Confidence: {(hifResult.confidence * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>

        {/* Fault Statistics */}
        <div className="card">
          <h2 className="card-header">Historical Statistics</h2>
          {faultHistory?.statistics && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-slate-700/50 rounded-lg">
                <p className="text-xs text-slate-400">Avg Restoration Time</p>
                <p className="text-xl font-bold text-white">
                  {faultHistory.statistics.avg_restoration_time_seconds}s
                </p>
              </div>
              <div className="p-3 bg-slate-700/50 rounded-lg">
                <p className="text-xs text-slate-400">Success Rate</p>
                <p className="text-xl font-bold text-green-400">
                  {faultHistory.statistics.success_rate_percent}%
                </p>
              </div>
              <div className="p-3 bg-slate-700/50 rounded-lg">
                <p className="text-xs text-slate-400">Most Common Fault</p>
                <p className="text-xl font-bold text-white">
                  {faultHistory.statistics.most_common_fault_type}
                </p>
              </div>
              <div className="p-3 bg-slate-700/50 rounded-lg">
                <p className="text-xs text-slate-400">Most Affected</p>
                <p className="text-xl font-bold text-white">
                  {faultHistory.statistics.most_affected_feeder}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Fault Types Reference */}
        <div className="card">
          <h2 className="card-header">Fault Types</h2>
          <div className="space-y-2">
            {[
              { type: 'LG', desc: 'Line-to-Ground (Single phase)' },
              { type: 'LL', desc: 'Line-to-Line (Two phase)' },
              { type: 'LLG', desc: 'Line-to-Line-to-Ground' },
              { type: '3PH', desc: 'Three-Phase (Most severe)' },
              { type: 'HIF', desc: 'High-Impedance Fault (Difficult to detect)' },
            ].map((fault) => (
              <div key={fault.type} className="flex items-center justify-between p-2 bg-slate-700/50 rounded">
                <span className="font-mono text-amber-400">{fault.type}</span>
                <span className="text-sm text-slate-400">{fault.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="card bg-slate-700/50">
        <p className="text-slate-400 text-sm">
          <span className="text-amber-400 font-medium">Note:</span> Currently displaying mock diagnostic results.
          The hybrid CNN-Transformer + R-GNN model will be integrated when training is complete.
          The model uses multi-modal data fusion (voltage + current signals) for comprehensive fault analysis.
        </p>
      </div>
    </div>
  );
}
