import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Square, Calendar, Clock, Zap,
  CheckCircle, XCircle, AlertTriangle, Loader2,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ComposedChart, Area,
  ReferenceLine,
} from 'recharts';
import api from '../services/api';
import { useGridStore } from '../stores/gridStore';
import { useSimulationPlayback } from '../hooks/useSimulationPlayback';
import type {
  PipelineTaskStatus, DayResult, SimulationStep, SingleDayResult, PipelineResults,
  GridState,
} from '../types';

type SimMode = 'single' | 'range';

export default function Simulation() {
  const { setGridState, setPipelineSteps } = useGridStore();
  const playback = useSimulationPlayback();

  // --- Mode & date state ---
  const [mode, setMode] = useState<SimMode>('single');
  const [singleDate, setSingleDate] = useState('2025-08-01');
  const [startDate, setStartDate] = useState('2025-08-01');
  const [endDate, setEndDate] = useState('2025-08-07');

  // --- Execution state ---
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<PipelineTaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // --- Results ---
  const [singleResult, setSingleResult] = useState<SingleDayResult | null>(null);
  const [multiResults, setMultiResults] = useState<DayResult[] | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- Cleanup polling on unmount ---
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // --- Poll for multi-day task progress ---
  const startPolling = useCallback((tid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getPipelineStatus(tid);
        setTaskStatus(status as unknown as PipelineTaskStatus);

        if (status.status === 'completed' || status.status === 'error' || status.status === 'cancelled') {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setRunning(false);

          if (status.status === 'completed') {
            const results = await api.getPipelineResults(tid);
            setMultiResults((results as unknown as PipelineResults).completed_days);
            // Fetch the grid state from the last simulation run
            try {
              const gs = await api.getGridStateNoSolve();
              setGridState(gs as unknown as GridState);
            } catch { /* model may not be loaded yet */ }
          }
          if (status.status === 'error') {
            setError(status.error || 'Simulation failed');
          }
        }
      } catch (e: unknown) {
        clearInterval(pollRef.current!);
        pollRef.current = null;
        setRunning(false);
        setError(e instanceof Error ? e.message : 'Polling failed');
      }
    }, 2000);
  }, [setGridState]);

  // --- Run simulation ---
  const handleRun = async () => {
    setError(null);
    setSingleResult(null);
    setMultiResults(null);
    setTaskStatus(null);
    setRunning(true);

    try {
      if (mode === 'single') {
        // Start playback on Dashboard (step-by-step real-time updates)
        playback.run(singleDate, singleDate);

        // Also fetch full results for this page's detailed charts
        const result = await api.simulateSingleDay(singleDate);
        setSingleResult(result as unknown as SingleDayResult);
        if ((result as any).grid_state) {
          setGridState((result as any).grid_state as GridState);
        }
        if ((result as any).steps) {
          setPipelineSteps(
            ((result as any).steps as any[]).map((s: any) => ({
              hour: s.hour ?? 0,
              total_power_kw: s.total_power_kw ?? 0,
              total_losses_kw: s.total_losses_kw ?? 0,
              min_voltage_pu: s.min_voltage_pu ?? 0,
              max_voltage_pu: s.max_voltage_pu ?? 0,
              voltage_violations: s.voltage_violations ?? 0,
              converged: s.converged ?? false,
            })),
            singleDate
          );
        }
        setRunning(false);
      } else {
        // For range mode, use playback for Dashboard + pipeline for this page's charts
        playback.run(startDate, endDate);

        const resp = await api.startPipelineSimulation(startDate, endDate);
        setTaskId(resp.task_id);
        startPolling(resp.task_id);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start simulation');
      setRunning(false);
    }
  };

  // --- Cancel ---
  const handleCancel = async () => {
    // Stop step-by-step playback on Dashboard
    playback.stop();

    if (taskId) {
      try {
        await api.cancelPipelineTask(taskId);
      } catch { /* ignore */ }
    }
    if (pollRef.current) clearInterval(pollRef.current);
    setRunning(false);
  };

  // --- Progress bar ---
  const progress = taskStatus
    ? Math.round((taskStatus.completed_count / Math.max(taskStatus.total_days, 1)) * 100)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center">
          <Zap className="w-6 h-6 mr-2 text-cyan-400" />
          Power System Simulation
        </h1>
        <p className="text-slate-400">
          Run real-data OpenDSS simulations for any date or date range (May-Aug 2025)
        </p>
      </div>

      {/* Controls card */}
      <div className="card">
        <h2 className="card-header flex items-center">
          <Calendar className="w-5 h-5 mr-2 text-blue-400" />
          Simulation Setup
        </h2>

        {/* Mode selector */}
        <div className="flex space-x-2 mb-4">
          <button
            onClick={() => setMode('single')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === 'single'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Single Day
          </button>
          <button
            onClick={() => setMode('range')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === 'range'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Date Range
          </button>
        </div>

        {/* Date inputs */}
        <div className="flex flex-wrap items-end gap-4 mb-4">
          {mode === 'single' ? (
            <div>
              <label className="block text-sm text-slate-400 mb-1">Date</label>
              <input
                type="date"
                value={singleDate}
                min="2025-05-01"
                max="2025-08-31"
                onChange={(e) => setSingleDate(e.target.value)}
                className="input-field"
                disabled={running}
              />
            </div>
          ) : (
            <>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  min="2025-05-01"
                  max="2025-08-31"
                  onChange={(e) => setStartDate(e.target.value)}
                  className="input-field"
                  disabled={running}
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  min={startDate}
                  max="2025-08-31"
                  onChange={(e) => setEndDate(e.target.value)}
                  className="input-field"
                  disabled={running}
                />
              </div>
              <div className="text-sm text-slate-400">
                {(() => {
                  const days = Math.floor(
                    (new Date(endDate).getTime() - new Date(startDate).getTime()) / 86400000
                  ) + 1;
                  return days > 0 ? `${days} day${days > 1 ? 's' : ''}` : '';
                })()}
              </div>
            </>
          )}

          {/* Run / Cancel button */}
          {!running ? (
            <button onClick={handleRun} className="btn-primary flex items-center">
              <Play className="w-4 h-4 mr-2" />
              Run Simulation
            </button>
          ) : (
            <button onClick={handleCancel} className="btn-danger flex items-center">
              <Square className="w-4 h-4 mr-2" />
              Cancel
            </button>
          )}
        </div>

        {/* Quick range presets */}
        {mode === 'range' && !running && (
          <div className="flex flex-wrap gap-2">
            <span className="text-xs text-slate-500 self-center mr-1">Quick:</span>
            {[
              { label: '1 Week', s: '2025-08-01', e: '2025-08-07' },
              { label: '2 Weeks', s: '2025-07-15', e: '2025-07-28' },
              { label: 'July', s: '2025-07-01', e: '2025-07-31' },
              { label: 'August', s: '2025-08-01', e: '2025-08-31' },
              { label: 'Full (May-Aug)', s: '2025-05-01', e: '2025-08-31' },
            ].map((p) => (
              <button
                key={p.label}
                onClick={() => { setStartDate(p.s); setEndDate(p.e); }}
                className="px-2 py-1 text-xs bg-slate-700 text-slate-300 rounded hover:bg-slate-600 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Progress bar (multi-day) */}
      {running && mode === 'range' && taskStatus && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <Loader2 className="w-4 h-4 mr-2 text-blue-400 animate-spin" />
              <span className="text-white text-sm font-medium">
                Day {taskStatus.current_day} / {taskStatus.total_days}
              </span>
            </div>
            <span className="text-slate-400 text-sm">
              {taskStatus.current_date} &mdash; {progress}%
            </span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-3">
            <div
              className="bg-blue-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Single-day loading spinner */}
      {running && mode === 'single' && (
        <div className="card flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin mr-3" />
          <span className="text-white">Running simulation for {singleDate}...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card bg-red-900/30 border-red-700">
          <div className="flex items-center">
            <XCircle className="w-5 h-5 text-red-400 mr-2 flex-shrink-0" />
            <span className="text-red-300">{error}</span>
          </div>
        </div>
      )}

      {/* ============ SINGLE-DAY RESULTS ============ */}
      {singleResult && <SingleDayResults result={singleResult} date={singleDate} />}

      {/* ============ MULTI-DAY RESULTS ============ */}
      {multiResults && <MultiDayResults results={multiResults} />}
    </div>
  );
}


// ================================================================
// SINGLE-DAY RESULTS COMPONENT
// ================================================================
function SingleDayResults({ result, date }: { result: SingleDayResult; date: string }) {
  const { summary, steps } = result;
  const convergedSteps = steps.filter((s) => s.converged);

  // Chart data: 96 steps at 15-min intervals
  const powerChart = steps.map((s) => ({
    time: `${String(Math.floor(s.hour)).padStart(2, '0')}:${String((s.hour % 1) * 60).padStart(2, '0')}`,
    hour: s.hour,
    power: s.total_power_kw ?? null,
    losses: s.total_losses_kw ?? null,
  }));

  const voltageChart = convergedSteps.map((s) => ({
    hour: s.hour,
    min_v: s.min_voltage_pu,
    max_v: s.max_voltage_pu,
    violations: s.voltage_violations,
  }));

  // Per-feeder power
  const feederChart = convergedSteps.map((s) => {
    const row: Record<string, unknown> = { hour: s.hour };
    for (const f of ['F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12']) {
      row[f] = s[`power_${f}_kw`] ?? null;
    }
    return row;
  });

  return (
    <>
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          label="Convergence"
          value={`${summary.converged_steps} / ${summary.total_steps}`}
          icon={<CheckCircle className="w-5 h-5 text-green-400" />}
          color={summary.converged_steps === 96 ? 'green' : 'amber'}
        />
        <SummaryCard
          label="Peak Power"
          value={`${((summary.peak_power_kw ?? 0) / 1000).toFixed(1)} MW`}
          icon={<Zap className="w-5 h-5 text-yellow-400" />}
        />
        <SummaryCard
          label="Voltage Range"
          value={`${summary.min_voltage_pu ?? '-'} - ${summary.max_voltage_pu ?? '-'} pu`}
          icon={<AlertTriangle className="w-5 h-5 text-amber-400" />}
        />
        <SummaryCard
          label="Date"
          value={date}
          icon={<Calendar className="w-5 h-5 text-blue-400" />}
        />
      </div>

      {/* Total power chart */}
      <div className="card">
        <h2 className="card-header">Total System Power (24h)</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={powerChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="hour"
                stroke="#9ca3af"
                tickFormatter={(h) => `${Math.floor(h)}:00`}
              />
              <YAxis stroke="#9ca3af" label={{ value: 'kW', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                formatter={(v: number, name: string) => [
                  `${(v / 1000).toFixed(2)} MW`,
                  name === 'power' ? 'Total Power' : 'Losses',
                ]}
                labelFormatter={(h) => `Hour ${Number(h).toFixed(2)}`}
              />
              <Line type="monotone" dataKey="power" stroke="#3b82f6" strokeWidth={2} dot={false} name="power" />
              <Line type="monotone" dataKey="losses" stroke="#ef4444" strokeWidth={1} dot={false} name="losses" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Voltage chart */}
      <div className="card">
        <h2 className="card-header">Voltage Profile (24h)</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={voltageChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" tickFormatter={(h) => `${Math.floor(h)}:00`} />
              <YAxis yAxisId="v" stroke="#9ca3af" domain={[0.7, 1.15]} label={{ value: 'pu', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="viol" orientation="right" stroke="#ef4444" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              />
              <ReferenceLine yAxisId="v" y={0.95} stroke="#ef4444" strokeDasharray="3 3" label="0.95" />
              <ReferenceLine yAxisId="v" y={1.05} stroke="#ef4444" strokeDasharray="3 3" label="1.05" />
              <Area yAxisId="v" type="monotone" dataKey="max_v" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} name="Max V" />
              <Area yAxisId="v" type="monotone" dataKey="min_v" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} name="Min V" />
              <Bar yAxisId="viol" dataKey="violations" fill="#ef444466" name="Violations" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-feeder power */}
      <div className="card">
        <h2 className="card-header">Per-Feeder Power (24h)</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={feederChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" tickFormatter={(h) => `${Math.floor(h)}:00`} />
              <YAxis stroke="#9ca3af" label={{ value: 'kW', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                formatter={(v: number) => [`${(v / 1000).toFixed(2)} MW`]}
              />
              {[
                { key: 'F06', color: '#3b82f6' },
                { key: 'F07', color: '#22c55e' },
                { key: 'F08', color: '#f59e0b' },
                { key: 'F09', color: '#ef4444' },
                { key: 'F10', color: '#8b5cf6' },
                { key: 'F11', color: '#ec4899' },
                { key: 'F12', color: '#06b6d4' },
              ].map((f) => (
                <Line
                  key={f.key}
                  type="monotone"
                  dataKey={f.key}
                  stroke={f.color}
                  strokeWidth={1.5}
                  dot={false}
                  name={f.key}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}


// ================================================================
// MULTI-DAY RESULTS COMPONENT
// ================================================================
function MultiDayResults({ results }: { results: DayResult[] }) {
  const successDays = results.filter((r) => r.status === 'success');
  const failedDays = results.filter((r) => r.status !== 'success');

  // Chart data
  const trendChart = results.map((r) => ({
    date: r.date.slice(5), // MM-DD
    fullDate: r.date,
    convergence: r.converged_steps ?? 0,
    peakMW: ((r.peak_power_kw ?? 0) / 1000),
    avgMW: ((r.avg_power_kw ?? 0) / 1000),
    minV: r.min_voltage_pu ?? 0,
    maxV: r.max_voltage_pu ?? 0,
    violations: r.total_violations ?? 0,
  }));

  // Stats
  const avgConvergence = successDays.length
    ? (successDays.reduce((a, r) => a + (r.converged_steps ?? 0), 0) / successDays.length).toFixed(1)
    : '0';
  const avgPeakMW = successDays.length
    ? (successDays.reduce((a, r) => a + ((r.peak_power_kw ?? 0) / 1000), 0) / successDays.length).toFixed(1)
    : '0';

  return (
    <>
      {/* Summary row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <SummaryCard label="Total Days" value={`${results.length}`} icon={<Calendar className="w-5 h-5 text-blue-400" />} />
        <SummaryCard label="Succeeded" value={`${successDays.length}`} icon={<CheckCircle className="w-5 h-5 text-green-400" />} color="green" />
        <SummaryCard label="Failed" value={`${failedDays.length}`} icon={<XCircle className="w-5 h-5 text-red-400" />} color={failedDays.length > 0 ? 'red' : 'green'} />
        <SummaryCard label="Avg Convergence" value={`${avgConvergence} / 96`} icon={<Clock className="w-5 h-5 text-cyan-400" />} />
        <SummaryCard label="Avg Peak Power" value={`${avgPeakMW} MW`} icon={<Zap className="w-5 h-5 text-yellow-400" />} />
      </div>

      {/* Power trend across days */}
      <div className="card">
        <h2 className="card-header">Daily Peak & Average Power</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={trendChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" angle={-45} textAnchor="end" height={50} tick={{ fontSize: 10 }} />
              <YAxis stroke="#9ca3af" label={{ value: 'MW', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                labelFormatter={(d) => `Date: ${d}`}
                formatter={(v: number) => [`${v.toFixed(1)} MW`]}
              />
              <Bar dataKey="peakMW" fill="#3b82f6" name="Peak MW" />
              <Bar dataKey="avgMW" fill="#22c55e80" name="Avg MW" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Voltage trend */}
      <div className="card">
        <h2 className="card-header">Daily Voltage Envelope</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={trendChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" angle={-45} textAnchor="end" height={50} tick={{ fontSize: 10 }} />
              <YAxis yAxisId="v" stroke="#9ca3af" domain={[0.7, 1.15]} />
              <YAxis yAxisId="viol" orientation="right" stroke="#ef4444" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              />
              <ReferenceLine yAxisId="v" y={0.95} stroke="#ef4444" strokeDasharray="3 3" />
              <ReferenceLine yAxisId="v" y={1.05} stroke="#ef4444" strokeDasharray="3 3" />
              <Line yAxisId="v" type="monotone" dataKey="maxV" stroke="#22c55e" strokeWidth={1.5} dot={false} name="Max V (pu)" />
              <Line yAxisId="v" type="monotone" dataKey="minV" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Min V (pu)" />
              <Bar yAxisId="viol" dataKey="violations" fill="#ef444444" name="Total Violations" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Convergence trend */}
      <div className="card">
        <h2 className="card-header">Daily Convergence</h2>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={trendChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" angle={-45} textAnchor="end" height={50} tick={{ fontSize: 10 }} />
              <YAxis stroke="#9ca3af" domain={[0, 96]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
              />
              <ReferenceLine y={96} stroke="#22c55e" strokeDasharray="3 3" label="96" />
              <Bar dataKey="convergence" name="Converged Steps" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Results table */}
      <div className="card overflow-x-auto">
        <h2 className="card-header">Per-Day Results</h2>
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 uppercase border-b border-slate-700">
            <tr>
              <th className="px-3 py-2">Date</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Conv.</th>
              <th className="px-3 py-2">Peak MW</th>
              <th className="px-3 py-2">Avg MW</th>
              <th className="px-3 py-2">Min V</th>
              <th className="px-3 py-2">Max V</th>
              <th className="px-3 py-2">Violations</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.date} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                <td className="px-3 py-2 text-white font-mono">{r.date}</td>
                <td className="px-3 py-2">
                  {r.status === 'success' ? (
                    <span className="text-green-400 flex items-center"><CheckCircle className="w-3 h-3 mr-1" /> OK</span>
                  ) : (
                    <span className="text-red-400 flex items-center"><XCircle className="w-3 h-3 mr-1" /> Fail</span>
                  )}
                </td>
                <td className="px-3 py-2 text-slate-300">{r.converged_steps ?? '-'}/96</td>
                <td className="px-3 py-2 text-slate-300">{r.peak_power_kw ? (r.peak_power_kw / 1000).toFixed(1) : '-'}</td>
                <td className="px-3 py-2 text-slate-300">{r.avg_power_kw ? (r.avg_power_kw / 1000).toFixed(1) : '-'}</td>
                <td className="px-3 py-2 text-slate-300">{r.min_voltage_pu ?? '-'}</td>
                <td className="px-3 py-2 text-slate-300">{r.max_voltage_pu ?? '-'}</td>
                <td className="px-3 py-2 text-slate-300">{r.total_violations ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}


// ================================================================
// REUSABLE SUMMARY CARD
// ================================================================
function SummaryCard({
  label, value, icon, color,
}: {
  label: string; value: string; icon: React.ReactNode; color?: string;
}) {
  const borderColor =
    color === 'green' ? 'border-green-700' :
    color === 'red' ? 'border-red-700' :
    color === 'amber' ? 'border-amber-700' :
    'border-slate-700';

  return (
    <div className={`card ${borderColor}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400 uppercase">{label}</span>
        {icon}
      </div>
      <p className="text-lg font-bold text-white">{value}</p>
    </div>
  );
}
