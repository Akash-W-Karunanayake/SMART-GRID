import { useEffect, useMemo, useState } from 'react';
import { BarChart3, RefreshCw, Battery, AlertCircle, CheckCircle } from 'lucide-react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Legend,
} from 'recharts';
import api from '../services/api';

type NetLoadPoint = { timestamp: string; yhat_mw: number };

// include run_id so we can fetch imbalance/actions from DB
type NetLoadForecastResponse = {
  target_date: string;
  issue_time: string;
  run_id?: number;
  computed_at?: string;
  points: NetLoadPoint[];
};

type ImbalanceItem = {
  step_idx: number;
  timestamp: string;
  yhat_mw: number;
  label: 'oversupply' | 'balanced' | 'undersupply';
  severity: number;
  recommended_action: string; // "Charge battery." / "Discharge battery." / "No action."
};

type NetLoadImbalanceResponse = {
  run_id: number;
  target_date: string;
  issue_time: string;
  threshold_X_mw: number;
  counts: { oversupply: number; balanced: number; undersupply: number };
  top_k: number;
  worst_undersupply: ImbalanceItem[];
  worst_oversupply: ImbalanceItem[];
  note?: string;
};

function fmtTime(tsISO: string) {
  const d = new Date(tsISO);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const fmtLocalDateTime = (iso: string) =>
  new Date(iso).toLocaleString([], {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

export default function NetLoad() {
  const [netLoadForecast, setNetLoadForecast] = useState<NetLoadForecastResponse | null>(null);
  const [imbalance, setImbalance] = useState<NetLoadImbalanceResponse | null>(null);

  // existing operator data section kept (optional)
  const [operatorData, setOperatorData] = useState<any>(null);

  const [loading, setLoading] = useState(false);

  // user chooses date; no auto-run
  const [targetDate, setTargetDate] = useState('2025-08-10');

  useEffect(() => {
  const appHeader = document.querySelector('header') as HTMLElement | null;

  if (appHeader) {
    appHeader.style.display = 'none';
  }

  return () => {
    if (appHeader) {
      appHeader.style.display = '';
    }
  };
  }, []);

  const runForecast = async () => {
    setLoading(true);
    try {
      // 1) run forecast (this also saves run + points in DB)
      const [netLoad, opData] = await Promise.all([
        api.forecastNetLoadByDate(targetDate) as Promise<NetLoadForecastResponse>,
        api.getGridOperatorData(),
      ]);

      setNetLoadForecast(netLoad);
      setOperatorData(opData);

      // 2) fetch imbalance/actions for that run_id (decision support)
      const runId = netLoad.run_id;
      if (runId) {
        const imb = (await api.getNetLoadImbalance(runId)) as NetLoadImbalanceResponse;
        setImbalance(imb);
      } else {
        setImbalance(null);
      }
    } catch (error) {
      console.error('Failed to run forecast:', error);
      setImbalance(null);
    } finally {
      setLoading(false);
    }
  };

  // --- KPIs computed from 96 points ---
  const kpis = useMemo(() => {
    const pts = netLoadForecast?.points ?? [];
    if (!pts.length) return null;

    let maxV = -Infinity,
      minV = Infinity;
    let maxT = pts[0]?.timestamp ?? '';
    let minT = pts[0]?.timestamp ?? '';
    let sum = 0;

    for (const p of pts) {
      const v = p.yhat_mw;
      sum += v;
      if (v > maxV) {
        maxV = v;
        maxT = p.timestamp;
      }
      if (v < minV) {
        minV = v;
        minT = p.timestamp;
      }
    }
    const avg = sum / pts.length;

    return {
      peakMW: maxV,
      peakTime: maxT,
      minMW: minV,
      minTime: minT,
      avgMW: avg,
    };
  }, [netLoadForecast]);

  // --- chart data + state classification based on imbalance threshold ---
  const thresholdX = imbalance?.threshold_X_mw ?? null;

  const chartData = useMemo(() => {
    const pts = netLoadForecast?.points ?? [];
    const X = thresholdX ?? 0;

    return pts.map((p) => {
      const state =
        thresholdX == null
          ? 'unknown'
          : p.yhat_mw > X
          ? 'undersupply'
          : p.yhat_mw < -X
          ? 'oversupply'
          : 'balanced';

      return {
        time: fmtTime(p.timestamp),
        netLoad: p.yhat_mw,
        state,
        timestamp: p.timestamp,
      };
    });
  }, [netLoadForecast, thresholdX]);

  // --- key-hours table: every 2 hours => indices 0, 8, 16, ... (8 steps = 2 hours) ---
  const keyHours = useMemo(() => {
    const pts = netLoadForecast?.points ?? [];
    if (!pts.length) return [];

    const X = thresholdX ?? 0;
    const rows = [];
    for (let i = 0; i < pts.length; i += 8) {
      const p = pts[i];
      const label =
        thresholdX == null
          ? '—'
          : p.yhat_mw > X
          ? 'Undersupply'
          : p.yhat_mw < -X
          ? 'Oversupply'
          : 'Balanced';

      const action =
        thresholdX == null
          ? '—'
          : label === 'Undersupply'
          ? 'Discharge battery.'
          : label === 'Oversupply'
          ? 'Charge battery.'
          : 'No action.';

      rows.push({
        time: fmtTime(p.timestamp),
        mw: p.yhat_mw,
        label,
        action,
      });
    }
    return rows;
  }, [netLoadForecast, thresholdX]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-purple-400" />
            Net Load Forecasting
          </h1>
          
          {netLoadForecast && (
            <p className="text-slate-400 text-sm mt-1">
              Target: <span className="text-white">{netLoadForecast.target_date}</span> • Issue time:{' '}
              <span className="text-white">{netLoadForecast.issue_time}</span>
              
              {netLoadForecast.computed_at ? (
                <>
                  {' '}
                  • Computed at:{' '}
                  <span className="text-white">{fmtLocalDateTime(netLoadForecast.computed_at)}</span>
                </>
                
              ) : null}
            </p>
          )}
        </div>

        </div>

      
      {/* Controls (button-only) */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-slate-400 text-sm">Target Date:</label>
        <input
          type="date"
          value={targetDate}
          onChange={(e) => setTargetDate(e.target.value)}
          className="input-field"
          min="2025-07-15"
          max="2025-08-31"
        />

        <button onClick={runForecast} disabled={loading} className="btn-primary flex items-center">
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Forecast
        </button>

        <span className="text-slate-500 text-xs">
          Select a date then click <span className="text-slate-300">Forecast</span>.
        </span>
      </div>

      {/* Tomorrow at a glance (KPIs) */}
      <div className="card">
        <h2 className="card-header">Tomorrow at a Glance</h2>
        {!kpis || !imbalance ? (
          <p className="text-slate-400 text-sm">Run a forecast to see summary metrics.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Peak MW</p>
              <p className="text-white text-lg font-semibold">{kpis.peakMW.toFixed(2)}</p>
              <p className="text-slate-400 text-xs">{fmtTime(kpis.peakTime)}</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Min MW</p>
              <p className="text-white text-lg font-semibold">{kpis.minMW.toFixed(2)}</p>
              <p className="text-slate-400 text-xs">{fmtTime(kpis.minTime)}</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Avg MW</p>
              <p className="text-white text-lg font-semibold">{kpis.avgMW.toFixed(2)}</p>
              <p className="text-slate-400 text-xs">Daily mean</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Undersupply</p>
              <p className="text-red-400 text-lg font-semibold">{imbalance.counts.undersupply}</p>
              <p className="text-slate-400 text-xs">Intervals</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Oversupply</p>
              <p className="text-blue-400 text-lg font-semibold">{imbalance.counts.oversupply}</p>
              <p className="text-slate-400 text-xs">Intervals</p>
            </div>
            <div className="p-3 bg-slate-700/50 rounded-lg">
              <p className="text-slate-400 text-xs">Balanced</p>
              <p className="text-green-400 text-lg font-semibold">{imbalance.counts.balanced}</p>
              <p className="text-slate-400 text-xs">Intervals</p>
            </div>
          </div>
        )}
      </div>

      {/* Key-hours table (every 2 hours) */}
      <div className="card">
        <h2 className="card-header">Key Hours (Every 2 Hours)</h2>
        {!netLoadForecast ? (
          <p className="text-slate-400 text-sm">Run a forecast to view key-hour values.</p>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-slate-400">
                <tr>
                  <th className="text-left py-2">Time</th>
                  <th className="text-left py-2">Forecast (MW)</th>
                  <th className="text-left py-2">State</th>
                  <th className="text-left py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {keyHours.map((r, idx) => (
                  <tr key={idx} className="border-t border-slate-700">
                    <td className="py-2 text-slate-200">{r.time}</td>
                    <td className="py-2 text-slate-200">{r.mw.toFixed(2)}</td>
                    <td className="py-2 text-slate-200">{r.label}</td>
                    <td className="py-2 text-slate-200">{r.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Main Chart */}
          
      <div className="card">
        <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            <h2 className="card-header mb-1">Net Load Forecast (Day-ahead)</h2>

            {netLoadForecast ? (
              <div className="flex flex-wrap items-center gap-3 mt-2">
                <span className="rounded-md border border-blue-500/30 bg-blue-500/10 px-3 py-1.5 text-sm text-blue-200 font-medium">
                  Forecast Date: {netLoadForecast.target_date}
                </span>
                <span className="text-slate-400 text-sm">
                  Net Load = Load Demand − Renewable Generation
                </span>
              </div>
            ) : (
              <p className="text-slate-400 text-sm">
                Net Load = Load Demand − Renewable Generation
              </p>
            )}
          </div>

          {thresholdX != null && (
            <div className="flex flex-wrap gap-2 text-xs">
              <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-slate-300">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-400" />
                Net Load
              </div>
              <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-slate-300">
                <span className="inline-block h-[2px] w-4 bg-red-500" />
                Undersupply Threshold (+{thresholdX.toFixed(2)} MW)
              </div>
              <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/60 px-3 py-1.5 text-slate-300">
                <span className="inline-block h-[2px] w-4 bg-blue-500" />
                Oversupply Threshold (-{thresholdX.toFixed(2)} MW)
              </div>
            </div>
          )}
        </div>

        <div className="mt-5 h-[340px] rounded-xl border border-slate-700/60 bg-slate-800/30 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 10, right: 12, left: 0, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={true} horizontal={true} />

              <XAxis
                dataKey="time"
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
                minTickGap={28}
                tickLine={false}
                axisLine={{ stroke: '#475569' }}
              />

              <YAxis
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: '#475569' }}
                width={48}
                label={{ value: 'MW', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '10px',
                  color: '#e2e8f0',
                }}
                labelStyle={{ color: '#cbd5e1', fontSize: 12 }}
                formatter={(value: any, name: string) => {
                  if (name === 'Net Load') return [`${Number(value).toFixed(2)} MW`, 'Net Load'];
                  return [`${Number(value).toFixed(2)} MW`, name];
                }}
                labelFormatter={(label) => `Time: ${label}`}
              />

              <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />

              {thresholdX != null && (
                <>
                  <ReferenceLine
                    y={thresholdX}
                    stroke="#ef4444"
                    strokeDasharray="4 4"
                  />
                  <ReferenceLine
                    y={-thresholdX}
                    stroke="#3b82f6"
                    strokeDasharray="4 4"
                  />
                </>
              )}

              <Line
                type="monotone"
                dataKey="netLoad"
                stroke="#8b5cf6"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 4 }}
                name="Net Load"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {netLoadForecast && (
          <div className="mt-3 flex items-center justify-center">
            <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1 text-xs text-slate-400">
              Forecast Date:{' '}
              <span className="text-slate-200 font-medium">
                {netLoadForecast.target_date}
              </span>
            </span>
          </div>
        )}

        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1">
            Positive net load = undersupply
          </span>
          <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1">
            Negative net load = oversupply
          </span>
          {imbalance?.note && (
            <span className="rounded-md border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 text-amber-300">
              Note: {imbalance.note}
            </span>
          )}
        </div>
      </div>

      

      {/* Top actions */}
      <div className="card">
        <h2 className="card-header">Recommended Actions (Top Intervals)</h2>
        {!imbalance ? (
          <p className="text-slate-400 text-sm">Run a forecast to generate actions.</p>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-white font-semibold mb-2">Top Undersupply</h3>
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead className="text-slate-400">
                    <tr>
                      <th className="text-left py-2">Time</th>
                      <th className="text-left py-2">MW</th>
                      <th className="text-left py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {imbalance.worst_undersupply.map((r, idx) => (
                      <tr key={idx} className="border-t border-slate-700">
                        <td className="py-2 text-slate-200">{fmtTime(r.timestamp)}</td>
                        <td className="py-2 text-slate-200">{r.yhat_mw.toFixed(2)}</td>
                        <td className="py-2 text-slate-200">{r.recommended_action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="text-white font-semibold mb-2">Top Oversupply</h3>
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead className="text-slate-400">
                    <tr>
                      <th className="text-left py-2">Time</th>
                      <th className="text-left py-2">MW</th>
                      <th className="text-left py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {imbalance.worst_oversupply.map((r, idx) => (
                      <tr key={idx} className="border-t border-slate-700">
                        <td className="py-2 text-slate-200">{fmtTime(r.timestamp)}</td>
                        <td className="py-2 text-slate-200">{r.yhat_mw.toFixed(2)}</td>
                        <td className="py-2 text-slate-200">{r.recommended_action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <p className="text-slate-400 text-xs lg:col-span-2">
              <span className="text-amber-400 font-medium">Note:</span> Actions assume battery storage is available.
            </p>
          </div>
        )}
      </div>

      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        

        
        

        
      </div>

      
    </div>
  );
}