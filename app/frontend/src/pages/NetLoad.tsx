import { useEffect, useMemo, useState } from 'react';
import { BarChart3, RefreshCw, Battery, Zap, ArrowDownToLine, ArrowUpFromLine } from 'lucide-react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
} from 'recharts';
import api from '../services/api';

type NetLoadPoint = { timestamp: string; yhat_mw: number };

type NetLoadForecastResponse = {
  target_date: string;
  issue_time: string;
  run_id?: number;
  computed_at?: string;
  points: NetLoadPoint[];
  actual_points?: NetLoadPoint[];
};

type ImbalanceItem = {
  step_idx: number;
  timestamp: string;
  yhat_mw: number;
  label: 'oversupply' | 'balanced' | 'undersupply';
  severity: number;
  recommended_action: string;
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
  timeline?: ImbalanceItem[];
  note?: string;
};

type ViewMode = 'priority' | 'full';
type Resolution = '15m' | '30m' | '1h' | '2h';
type StateFilter = 'all' | 'undersupply' | 'oversupply' | 'balanced';
type ChartMode = 'forecast' | 'compare';

function fmtTime(tsISO: string) {
  const d = new Date(tsISO);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

const fmtDisplayDateTime = (iso: string) =>
  new Date(iso).toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });

function stateBadgeClass(label: string) {
  const x = label.toLowerCase();
  if (x === 'undersupply') return 'border-red-500/20 bg-red-500/10 text-red-300';
  if (x === 'oversupply') return 'border-blue-500/20 bg-blue-500/10 text-blue-300';
  if (x === 'balanced') return 'border-green-500/20 bg-green-500/10 text-green-300';
  return 'border-slate-600 bg-slate-800/50 text-slate-300';
}

function actionBadgeClass(action: string) {
  const x = action.toLowerCase();
  if (x.includes('discharge')) return 'border-red-500/20 bg-red-500/10 text-red-300';
  if (x.includes('charge')) return 'border-blue-500/20 bg-blue-500/10 text-blue-300';
  return 'border-green-500/20 bg-green-500/10 text-green-300';
}

function getResolutionStep(resolution: Resolution) {
  switch (resolution) {
    case '15m':
      return 1;
    case '30m':
      return 2;
    case '1h':
      return 4;
    case '2h':
      return 8;
    default:
      return 4;
  }
}

function resolutionLabel(resolution: Resolution) {
  switch (resolution) {
    case '15m':
      return '15 min';
    case '30m':
      return '30 min';
    case '1h':
      return '1 hour';
    case '2h':
      return '2 hours';
    default:
      return '1 hour';
  }
}

export default function NetLoad() {
  const [netLoadForecast, setNetLoadForecast] = useState<NetLoadForecastResponse | null>(null);
  const [imbalance, setImbalance] = useState<NetLoadImbalanceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [targetDate, setTargetDate] = useState('2025-08-10');

  const [viewMode, setViewMode] = useState<ViewMode>('priority');
  const [fullDayResolution, setFullDayResolution] = useState<Resolution>('1h');
  const [fullDayStateFilter, setFullDayStateFilter] = useState<StateFilter>('all');

  const [chartMode, setChartMode] = useState<ChartMode>('forecast');
  const [actualSeries, setActualSeries] = useState<NetLoadPoint[]>([]);

  useEffect(() => {
    const appHeader = document.querySelector('header') as HTMLElement | null;
    if (appHeader) appHeader.style.display = 'none';

    return () => {
      if (appHeader) appHeader.style.display = '';
    };
  }, []);

  const runForecast = async () => {
    setLoading(true);
    try {
      const netLoad = (await api.forecastNetLoadByDate(targetDate)) as NetLoadForecastResponse;
      setNetLoadForecast(netLoad);
      setActualSeries(netLoad.actual_points ?? []);

      const runId = netLoad.run_id;
      if (runId) {
        const imb = (await api.getNetLoadImbalance(runId)) as NetLoadImbalanceResponse;
        setImbalance(imb);
      } else {
        setImbalance(null);
      }

      if (!(netLoad.actual_points && netLoad.actual_points.length > 0)) {
        setChartMode('forecast');
      }
    } catch (error) {
      console.error('Failed to run forecast:', error);
      setImbalance(null);
      setActualSeries([]);
      setChartMode('forecast');
    } finally {
      setLoading(false);
    }
  };

  const kpis = useMemo(() => {
    const pts = netLoadForecast?.points ?? [];
    if (!pts.length) return null;

    let maxV = -Infinity;
    let minV = Infinity;
    let maxT = pts[0]?.timestamp ?? '';
    let minT = pts[0]?.timestamp ?? '';

    for (const p of pts) {
      const v = p.yhat_mw;
      if (v > maxV) {
        maxV = v;
        maxT = p.timestamp;
      }
      if (v < minV) {
        minV = v;
        minT = p.timestamp;
      }
    }

    return {
      peakMW: maxV,
      peakTime: maxT,
      minMW: minV,
      minTime: minT,
    };
  }, [netLoadForecast]);

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

  const topUndersupply = useMemo(() => imbalance?.worst_undersupply.slice(0, 5) ?? [], [imbalance]);

  const topOversupply = useMemo(() => imbalance?.worst_oversupply.slice(0, 5) ?? [], [imbalance]);

  const actionSummary = useMemo(() => {
    if (!imbalance) return null;
    return {
      worstUnder: imbalance.worst_undersupply[0],
      worstOver: imbalance.worst_oversupply[0],
    };
  }, [imbalance]);

  const insightText = useMemo(() => {
    if (!imbalance || !kpis) return null;

    const over = imbalance.counts.oversupply;
    const under = imbalance.counts.undersupply;

    if (under > over) {
      return `The selected day shows stronger undersupply pressure than oversupply, with the highest risk around ${fmtTime(
        kpis.peakTime
      )}. Battery discharge is most relevant during the higher net-load period.`;
    }

    if (over > under) {
      return `The selected day shows stronger oversupply conditions than undersupply, with the lowest net load around ${fmtTime(
        kpis.minTime
      )}. Battery charging is most relevant around the low net-load period.`;
    }

    return `The selected day contains a mixed pattern of oversupply and undersupply intervals. Storage actions should be timed around threshold crossings.`;
  }, [imbalance, kpis]);

  const fullDayRows = useMemo(() => {
    const timeline = imbalance?.timeline ?? [];
    const step = getResolutionStep(fullDayResolution);
    const sampled = timeline.filter((_, idx) => idx % step === 0);

    if (fullDayStateFilter === 'all') return sampled;
    return sampled.filter((row) => row.label === fullDayStateFilter);
  }, [imbalance, fullDayResolution, fullDayStateFilter]);

  const hasActualData = actualSeries.length > 0;

  const comparisonChartData = useMemo(() => {
    const forecastPts = netLoadForecast?.points ?? [];
    const actualMap = new Map(actualSeries.map((p) => [p.timestamp, p.yhat_mw]));

    return forecastPts.map((p) => ({
      time: fmtTime(p.timestamp),
      timestamp: p.timestamp,
      predicted: p.yhat_mw,
      actual: actualMap.get(p.timestamp) ?? null,
    }));
  }, [netLoadForecast, actualSeries]);

  return (
    <div className="space-y-10">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-purple-400" />
            Net Load Forecasting
          </h1>
        </div>

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
        </div>
      </div>

      <div className="card mt-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="card-header">Forecast Summary</h2>
          <span className="text-xs text-slate-500">Operational overview</span>
        </div>

        {!kpis || !imbalance || !netLoadForecast ? (
          <p className="text-slate-400 text-sm">Run a forecast to see the summary view.</p>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap gap-3 text-sm">
              <span className="rounded-md border border-blue-500/20 bg-blue-500/10 px-3 py-1.5 text-blue-200">
                Target Date: <span className="font-medium">{netLoadForecast.target_date}</span>
              </span>
              <span className="rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                Issue Time: <span className="text-white">{fmtDisplayDateTime(netLoadForecast.issue_time)}</span>
              </span>
              {netLoadForecast.computed_at && (
                <span className="rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                  Computed At: <span className="text-white">{fmtDisplayDateTime(netLoadForecast.computed_at)}</span>
                </span>
              )}
              {thresholdX != null && (
                <span className="rounded-md border border-violet-500/20 bg-violet-500/10 px-3 py-1.5 text-violet-200">
                  Threshold: ±{thresholdX.toFixed(2)} MW
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
              <div className="rounded-xl border border-slate-700 bg-slate-800/40 p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide">Peak Net Load</p>
                <p className="mt-2 text-white text-2xl font-semibold">{kpis.peakMW.toFixed(2)} MW</p>
                <p className="text-slate-400 text-sm mt-1">{fmtTime(kpis.peakTime)}</p>
              </div>

              <div className="rounded-xl border border-slate-700 bg-slate-800/40 p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide">Minimum Net Load</p>
                <p className="mt-2 text-white text-2xl font-semibold">{kpis.minMW.toFixed(2)} MW</p>
                <p className="text-slate-400 text-sm mt-1">{fmtTime(kpis.minTime)}</p>
              </div>

              <div className="rounded-xl border border-red-500/10 bg-red-500/5 p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide">Undersupply Intervals</p>
                <p className="mt-2 text-red-300 text-2xl font-semibold">{imbalance.counts.undersupply}</p>
                <p className="text-slate-400 text-sm mt-1">Above + threshold</p>
              </div>

              <div className="rounded-xl border border-blue-500/10 bg-blue-500/5 p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide">Oversupply Intervals</p>
                <p className="mt-2 text-blue-300 text-2xl font-semibold">{imbalance.counts.oversupply}</p>
                <p className="text-slate-400 text-sm mt-1">Below - threshold</p>
              </div>

              <div className="rounded-xl border border-green-500/10 bg-green-500/5 p-4">
                <p className="text-slate-400 text-xs uppercase tracking-wide">Balanced Intervals</p>
                <p className="mt-2 text-green-300 text-2xl font-semibold">{imbalance.counts.balanced}</p>
                <p className="text-slate-400 text-sm mt-1">Within threshold band</p>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="card">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3 mb-1">
              <h2 className="card-header mb-0">Day-Ahead Net Load Forecast</h2>

              {netLoadForecast && (
                <span className="rounded-md border border-blue-500/20 bg-blue-500/10 px-3 py-1.5 text-sm text-blue-200 font-medium">
                  {netLoadForecast.target_date}
                </span>
              )}
            </div>

            <p className="text-slate-400 text-sm">
              {chartMode === 'forecast'
                ? '15-minute forecast • Net Load = Electricity Demand − Renewable Generation'
                : 'Predicted vs actual net load comparison for the selected day'}
            </p>
          </div>

          <div className="flex flex-col items-start gap-3 lg:min-w-[420px]">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setChartMode('forecast')}
                className={`rounded-md border px-3 py-1.5 text-sm transition ${
                  chartMode === 'forecast'
                    ? 'border-violet-500/30 bg-violet-500/10 text-violet-200'
                    : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:bg-slate-800'
                }`}
              >
                Forecast
              </button>

              <button
                type="button"
                onClick={() => hasActualData && setChartMode('compare')}
                disabled={!hasActualData}
                className={`rounded-md border px-3 py-1.5 text-sm transition ${
                  chartMode === 'compare'
                    ? 'border-violet-500/30 bg-violet-500/10 text-violet-200'
                    : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:bg-slate-800'
                } ${!hasActualData ? 'cursor-not-allowed opacity-50' : ''}`}
              >
                Predicted vs Actual
              </button>
            </div>

            <div className="min-h-[60px]">
              {!hasActualData && (
                <p className="text-xs text-slate-500 mb-2">Actual values not available for this date.</p>
              )}

              {chartMode === 'forecast' && thresholdX != null && (
                <div className="flex flex-wrap gap-2 text-xs">
                  <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-400" />
                    Net Load
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                    <span className="inline-block h-[2px] w-4 bg-red-500" />
                    Undersupply (+{thresholdX.toFixed(2)} MW)
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                    <span className="inline-block h-[2px] w-4 bg-blue-500" />
                    Oversupply (-{thresholdX.toFixed(2)} MW)
                  </div>
                </div>
              )}

              {chartMode === 'compare' && (
                <div className="flex flex-wrap gap-2 text-xs">
                  <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-400" />
                    Predicted
                  </div>
                  <div className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/50 px-3 py-1.5 text-slate-300">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    Actual
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-5 h-[380px] rounded-xl border border-slate-700/50 bg-slate-800/20 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartMode === 'forecast' ? chartData : comparisonChartData}
              margin={{ top: 10, right: 10, left: 0, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={true} horizontal={true} />

              <XAxis
                dataKey="time"
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
                minTickGap={40}
                tickLine={false}
                axisLine={{ stroke: '#475569' }}
              />

              <YAxis
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: '#475569' }}
                width={50}
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
                formatter={(value: number | null, name: string) => {
                  if (value == null) return ['N/A', name];
                  return [`${Number(value).toFixed(2)} MW`, name];
                }}
                labelFormatter={(label) => `Time: ${label}`}
              />

              <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />

              {chartMode === 'forecast' && thresholdX != null && (
                <>
                  <ReferenceLine y={thresholdX} stroke="#ef4444" strokeDasharray="4 4" />
                  <ReferenceLine y={-thresholdX} stroke="#3b82f6" strokeDasharray="4 4" />
                </>
              )}

              {chartMode === 'forecast' ? (
                <Line
                  type="monotone"
                  dataKey="netLoad"
                  stroke="#8b5cf6"
                  strokeWidth={3}
                  dot={false}
                  activeDot={{ r: 4 }}
                  name="Net Load"
                />
              ) : (
                <>
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#8b5cf6"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 4 }}
                    name="Predicted"
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#34d399"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 4 }}
                    name="Actual"
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3 text-xs">
          {chartMode === 'forecast' ? (
            <>
              <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1 text-slate-300">
                Positive net load = undersupply
              </span>
              <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1 text-slate-300">
                Negative net load = oversupply
              </span>
            </>
          ) : (
            <span className="rounded-md border border-slate-700 bg-slate-800/50 px-2.5 py-1 text-slate-300">
              Comparison view is shown only when actual values are available
            </span>
          )}
        </div>
      </div>

      <div className="card">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="card-header">Action Detail View</h2>
            <p className="text-slate-400 text-sm mt-1">
              Switch between a quick priority summary and the full daily action timeline.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setViewMode('priority')}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                viewMode === 'priority'
                  ? 'border-violet-500/30 bg-violet-500/10 text-violet-200'
                  : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:bg-slate-800'
              }`}
            >
              Priority View
            </button>
            <button
              type="button"
              onClick={() => setViewMode('full')}
              className={`rounded-md border px-3 py-1.5 text-sm transition ${
                viewMode === 'full'
                  ? 'border-violet-500/30 bg-violet-500/10 text-violet-200'
                  : 'border-slate-700 bg-slate-800/50 text-slate-300 hover:bg-slate-800'
              }`}
            >
              Full Day View
            </button>
          </div>
        </div>
      </div>

      {viewMode === 'priority' ? (
        <>
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Battery className="w-5 h-5 text-emerald-400" />
              <h2 className="card-header">Action Summary</h2>
            </div>

            {!actionSummary ? (
              <p className="text-slate-400 text-sm">Run a forecast to view action priorities.</p>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <ArrowUpFromLine className="w-4 h-4 text-red-400" />
                    <p className="text-red-300 font-medium">Highest Undersupply Risk</p>
                  </div>
                  {actionSummary.worstUnder ? (
                    <>
                      <p className="text-white text-lg font-semibold">{fmtTime(actionSummary.worstUnder.timestamp)}</p>
                      <p className="text-slate-300 text-sm mt-1">{actionSummary.worstUnder.yhat_mw.toFixed(2)} MW</p>
                      <span
                        className={`mt-3 inline-flex rounded-md border px-2.5 py-1 text-xs ${actionBadgeClass(
                          actionSummary.worstUnder.recommended_action
                        )}`}
                      >
                        {actionSummary.worstUnder.recommended_action}
                      </span>
                    </>
                  ) : (
                    <p className="text-slate-400 text-sm">No major undersupply interval.</p>
                  )}
                </div>

                <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <ArrowDownToLine className="w-4 h-4 text-blue-400" />
                    <p className="text-blue-300 font-medium">Highest Oversupply Risk</p>
                  </div>
                  {actionSummary.worstOver ? (
                    <>
                      <p className="text-white text-lg font-semibold">{fmtTime(actionSummary.worstOver.timestamp)}</p>
                      <p className="text-slate-300 text-sm mt-1">{actionSummary.worstOver.yhat_mw.toFixed(2)} MW</p>
                      <span
                        className={`mt-3 inline-flex rounded-md border px-2.5 py-1 text-xs ${actionBadgeClass(
                          actionSummary.worstOver.recommended_action
                        )}`}
                      >
                        {actionSummary.worstOver.recommended_action}
                      </span>
                    </>
                  ) : (
                    <p className="text-slate-400 text-sm">No major oversupply interval.</p>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <h2 className="card-header">Priority Actions</h2>
              <span className="text-xs text-slate-500">Top 5 intervals</span>
            </div>

            {!imbalance ? (
              <p className="text-slate-400 text-sm">Run a forecast to generate actions.</p>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-white font-semibold mb-3">Top Undersupply</h3>
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
                        {topUndersupply.map((r, idx) => (
                          <tr key={idx} className="border-t border-slate-700">
                            <td className="py-3 text-slate-200">{fmtTime(r.timestamp)}</td>
                            <td className="py-3 text-slate-200 font-medium">{r.yhat_mw.toFixed(2)}</td>
                            <td className="py-3">
                              <span className={`inline-flex rounded-md border px-2.5 py-1 text-xs ${actionBadgeClass(r.recommended_action)}`}>
                                {r.recommended_action}
                              </span>
                            </td>
                          </tr>
                        ))}
                        {!topUndersupply.length && (
                          <tr>
                            <td colSpan={3} className="py-3 text-slate-400">
                              No undersupply intervals above threshold.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div>
                  <h3 className="text-white font-semibold mb-3">Top Oversupply</h3>
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
                        {topOversupply.map((r, idx) => (
                          <tr key={idx} className="border-t border-slate-700">
                            <td className="py-3 text-slate-200">{fmtTime(r.timestamp)}</td>
                            <td className="py-3 text-slate-200 font-medium">{r.yhat_mw.toFixed(2)}</td>
                            <td className="py-3">
                              <span className={`inline-flex rounded-md border px-2.5 py-1 text-xs ${actionBadgeClass(r.recommended_action)}`}>
                                {r.recommended_action}
                              </span>
                            </td>
                          </tr>
                        ))}
                        {!topOversupply.length && (
                          <tr>
                            <td colSpan={3} className="py-3 text-slate-400">
                              No oversupply intervals below threshold.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-5 h-5 text-amber-400" />
              <h2 className="card-header">Forecast Insight</h2>
            </div>

            {!insightText ? (
              <p className="text-slate-400 text-sm">Run a forecast to generate operational insight.</p>
            ) : (
              <p className="text-slate-300 leading-7">{insightText}</p>
            )}
          </div>
        </>
      ) : (
        <div className="card">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between mb-4">
            <div>
              <h2 className="card-header">Full Day Action Timeline</h2>
              <p className="text-slate-400 text-sm mt-1">
                Showing the daily action profile at {resolutionLabel(fullDayResolution)} resolution.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="text-slate-400 text-sm">Resolution:</label>
                <select
                  value={fullDayResolution}
                  onChange={(e) => setFullDayResolution(e.target.value as Resolution)}
                  className="input-field"
                >
                  <option value="15m">15 min</option>
                  <option value="30m">30 min</option>
                  <option value="1h">1 hour</option>
                  <option value="2h">2 hours</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-slate-400 text-sm">State:</label>
                <select
                  value={fullDayStateFilter}
                  onChange={(e) => setFullDayStateFilter(e.target.value as StateFilter)}
                  className="input-field"
                >
                  <option value="all">All</option>
                  <option value="undersupply">Undersupply</option>
                  <option value="oversupply">Oversupply</option>
                  <option value="balanced">Balanced</option>
                </select>
              </div>
            </div>
          </div>

          {!imbalance ? (
            <p className="text-slate-400 text-sm">Run a forecast to view the full daily action timeline.</p>
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
                  {fullDayRows.map((r, idx) => (
                    <tr key={`${r.step_idx}-${idx}`} className="border-t border-slate-700">
                      <td className="py-3 text-slate-200">{fmtTime(r.timestamp)}</td>
                      <td className="py-3 text-slate-200">{r.yhat_mw.toFixed(2)}</td>
                      <td className="py-3">
                        <span className={`inline-flex rounded-md border px-2.5 py-1 text-xs ${stateBadgeClass(r.label)}`}>
                          {r.label}
                        </span>
                      </td>
                      <td className="py-3">
                        <span className={`inline-flex rounded-md border px-2.5 py-1 text-xs ${actionBadgeClass(r.recommended_action)}`}>
                          {r.recommended_action}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {!fullDayRows.length && (
                    <tr>
                      <td colSpan={4} className="py-3 text-slate-400">
                        No action timeline data available for the selected filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}