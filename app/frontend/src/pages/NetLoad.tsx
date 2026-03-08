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

  if (x === 'undersupply') {
    return 'border-red-400/35 bg-red-900/35 text-red-200 shadow-sm';
  }

  if (x === 'oversupply') {
    return 'border-blue-400/35 bg-blue-900/35 text-blue-200 shadow-sm';
  }

  if (x === 'balanced') {
    return 'border-green-400/30 bg-green-900/30 text-green-200 shadow-sm';
  }

  return 'border-slate-500/40 bg-slate-800/80 text-slate-200';
}

function actionBadgeClass(action: string) {
  const x = action.toLowerCase();

  if (x.includes('discharge')) {
    return 'border-red-400/30 bg-red-900/25 text-red-200 shadow-sm';
  }

  if (x.includes('charge')) {
    return 'border-blue-400/30 bg-blue-900/25 text-blue-200 shadow-sm';
  }

  if (x.includes('monitor') || x.includes('hold') || x.includes('stable') || x.includes('no action')) {
    return 'border-green-400/25 bg-green-900/20 text-green-200 shadow-sm';
  }

  return 'border-emerald-400/25 bg-emerald-900/20 text-emerald-200 shadow-sm';
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

function roundUp(value: number, step = 2) {
  return Math.ceil(value / step) * step;
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

  const forecastYAxisDomain = useMemo<[number, number]>(() => {
    return [-8, 12];
  }, []);

  const compareYAxisDomain = useMemo<[number, number]>(() => {
    const vals = comparisonChartData
      .flatMap((d) => [d.predicted, d.actual])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));

    if (!vals.length) return [-10, 10];

    const absMax = Math.max(...vals.map((v) => Math.abs(v)));
    const padded = Math.max(2, absMax * 1.12);
    const bound = roundUp(padded, 2);

    return [-bound, bound];
  }, [comparisonChartData]);

  const activeYAxisDomain = chartMode === 'forecast' ? forecastYAxisDomain : compareYAxisDomain;

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="flex items-center text-3xl font-bold tracking-tight text-white">
            <BarChart3 className="mr-2 h-6 w-6 text-violet-400" />
            Net Load Forecasting
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-600/70 bg-slate-800/80 px-4 py-3 shadow-lg">
          <label className="text-sm font-medium text-slate-200">Target Date:</label>
          <input
            type="date"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            className="input-field border-slate-500 bg-slate-900/90 text-slate-100 [color-scheme:dark]"
            min="2025-07-15"
            max="2025-08-31"
          />
          <button
            onClick={runForecast}
            disabled={loading}
            className="btn-primary flex items-center border border-violet-400/30 bg-violet-600/90 text-white shadow-md hover:bg-violet-500 disabled:opacity-60"
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Forecast
          </button>
        </div>
      </div>

      <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="card-header text-white">Forecast Summary</h2>
          <span className="text-xs font-medium text-slate-300">Operational overview</span>
        </div>

        {!kpis || !imbalance || !netLoadForecast ? (
          <p className="text-sm text-slate-300">Run a forecast to see the summary view.</p>
        ) : (
          <>
            <div className="mb-5 flex flex-wrap gap-3 text-sm">
              <span className="rounded-lg border border-blue-400/30 bg-blue-900/30 px-3 py-1.5 text-blue-100 shadow-sm">
                Target Date: <span className="font-semibold">{netLoadForecast.target_date}</span>
              </span>

              <span className="rounded-lg border border-slate-500/50 bg-slate-900/80 px-3 py-1.5 text-slate-200 shadow-sm">
                Issue Time: <span className="font-medium text-white">{fmtDisplayDateTime(netLoadForecast.issue_time)}</span>
              </span>

              {netLoadForecast.computed_at && (
                <span className="rounded-lg border border-slate-500/50 bg-slate-900/80 px-3 py-1.5 text-slate-200 shadow-sm">
                  Computed At: <span className="font-medium text-white">{fmtDisplayDateTime(netLoadForecast.computed_at)}</span>
                </span>
              )}

              {thresholdX != null && (
                <span className="rounded-lg border border-violet-400/30 bg-violet-900/25 px-3 py-1.5 text-violet-100 shadow-sm">
                  Threshold: ±{thresholdX.toFixed(2)} MW
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
              <div className="rounded-2xl border border-slate-500/50 bg-slate-900/80 p-4 shadow-md">
                <p className="text-xs uppercase tracking-wide text-slate-300">Peak Net Load</p>
                <p className="mt-2 text-2xl font-semibold text-white">{kpis.peakMW.toFixed(2)} MW</p>
                <p className="mt-1 text-sm text-slate-300">{fmtTime(kpis.peakTime)}</p>
              </div>

              <div className="rounded-2xl border border-slate-500/50 bg-slate-900/80 p-4 shadow-md">
                <p className="text-xs uppercase tracking-wide text-slate-300">Minimum Net Load</p>
                <p className="mt-2 text-2xl font-semibold text-white">{kpis.minMW.toFixed(2)} MW</p>
                <p className="mt-1 text-sm text-slate-300">{fmtTime(kpis.minTime)}</p>
              </div>

              <div className="rounded-2xl border border-red-400/35 bg-red-900/35 p-4 shadow-md">
                <p className="text-xs uppercase tracking-wide text-red-100/90">Undersupply Intervals</p>
                <p className="mt-2 text-2xl font-semibold text-red-100">{imbalance.counts.undersupply}</p>
                <p className="mt-1 text-sm text-red-100/80">Above + threshold</p>
              </div>

              <div className="rounded-2xl border border-blue-400/35 bg-blue-900/35 p-4 shadow-md">
                <p className="text-xs uppercase tracking-wide text-blue-100/90">Oversupply Intervals</p>
                <p className="mt-2 text-2xl font-semibold text-blue-100">{imbalance.counts.oversupply}</p>
                <p className="mt-1 text-sm text-blue-100/80">Below - threshold</p>
              </div>

              <div className="rounded-2xl border border-green-400/30 bg-green-900/30 p-4 shadow-md">
                <p className="text-xs uppercase tracking-wide text-green-100/90">Balanced Intervals</p>
                <p className="mt-2 text-2xl font-semibold text-green-100">{imbalance.counts.balanced}</p>
                <p className="mt-1 text-sm text-green-100/80">Within threshold band</p>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-3">
              <h2 className="card-header mb-0 text-white">Day-Ahead Net Load Forecast</h2>

              {netLoadForecast && (
                <span className="rounded-lg border border-blue-400/30 bg-blue-900/30 px-3 py-1.5 text-sm font-semibold text-blue-100 shadow-sm">
                  {netLoadForecast.target_date}
                </span>
              )}
            </div>

            <p className="text-sm text-slate-300">
              {chartMode === 'forecast'
                ? '15-minute forecast • Net Load = Electricity Demand − Renewable Generation'
                : 'Predicted vs actual net load comparison for the selected day'}
            </p>
          </div>

          <div className="flex flex-col items-start gap-3 lg:min-w-[430px]">
            <div className="flex items-center gap-2 rounded-xl border border-slate-600/60 bg-slate-900/75 p-1 shadow-sm">
              <button
                type="button"
                onClick={() => setChartMode('forecast')}
                className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
                  chartMode === 'forecast'
                    ? 'border-violet-400/30 bg-violet-900/30 text-violet-100 shadow-sm'
                    : 'border-slate-600 bg-slate-800/80 text-slate-200 hover:bg-slate-700'
                }`}
              >
                Forecast
              </button>

              <button
                type="button"
                onClick={() => hasActualData && setChartMode('compare')}
                disabled={!hasActualData}
                className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
                  chartMode === 'compare'
                    ? 'border-violet-400/30 bg-violet-900/30 text-violet-100 shadow-sm'
                    : 'border-slate-600 bg-slate-800/80 text-slate-200 hover:bg-slate-700'
                } ${!hasActualData ? 'cursor-not-allowed opacity-50' : ''}`}
              >
                Predicted vs Actual
              </button>
            </div>

            <div className="min-h-[60px]">
              {!hasActualData && (
                <p className="mb-2 text-xs text-slate-300">Actual values not available for this date.</p>
              )}

              {chartMode === 'forecast' && thresholdX != null && (
                <div className="flex flex-wrap gap-2 text-xs">
                  <div className="flex items-center gap-2 rounded-lg border border-slate-500/50 bg-slate-900/80 px-3 py-1.5 text-slate-200 shadow-sm">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-400" />
                    Net Load
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border border-red-400/30 bg-red-900/20 px-3 py-1.5 text-red-100 shadow-sm">
                    <span className="inline-block h-[2px] w-4 bg-red-400" />
                    Undersupply (+{thresholdX.toFixed(2)} MW)
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border border-blue-400/30 bg-blue-900/20 px-3 py-1.5 text-blue-100 shadow-sm">
                    <span className="inline-block h-[2px] w-4 bg-blue-400" />
                    Oversupply (-{thresholdX.toFixed(2)} MW)
                  </div>
                </div>
              )}

              {chartMode === 'compare' && (
                <div className="flex flex-wrap gap-2 text-xs">
                  <div className="flex items-center gap-2 rounded-lg border border-slate-500/50 bg-slate-900/80 px-3 py-1.5 text-slate-200 shadow-sm">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-400" />
                    Predicted
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border border-emerald-400/30 bg-emerald-900/20 px-3 py-1.5 text-emerald-100 shadow-sm">
                    <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    Actual
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-5 h-[390px] rounded-2xl border border-slate-600/70 bg-slate-950/65 p-4 shadow-inner">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartMode === 'forecast' ? chartData : comparisonChartData}
              margin={{ top: 10, right: 12, left: 4, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={true} horizontal={true} />

              <XAxis
                dataKey="time"
                stroke="#cbd5e1"
                tick={{ fontSize: 11 }}
                minTickGap={42}
                tickLine={false}
                axisLine={{ stroke: '#64748b' }}
              />

              <YAxis
                domain={activeYAxisDomain}
                allowDataOverflow={false}
                stroke="#cbd5e1"
                tick={{ fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: '#64748b' }}
                width={56}
                ticks={chartMode === 'forecast' ? [-8, -4, 0, 4, 8, 12] : undefined}
                label={{ value: 'MW', angle: -90, position: 'insideLeft', fill: '#cbd5e1' }}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #475569',
                  borderRadius: '12px',
                  color: '#e2e8f0',
                  boxShadow: '0 12px 28px rgba(0,0,0,0.35)',
                }}
                labelStyle={{ color: '#e2e8f0', fontSize: 12 }}
                formatter={(value: number | null, name: string) => {
                  if (value == null) return ['N/A', name];
                  return [`${Number(value).toFixed(2)} MW`, name];
                }}
                labelFormatter={(_, payload) => {
                  const ts = payload?.[0]?.payload?.timestamp;
                  return ts ? fmtDisplayDateTime(ts) : '';
                }}
              />

              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />

              {chartMode === 'forecast' && thresholdX != null && (
                <>
                  <ReferenceLine y={thresholdX} stroke="#f87171" strokeDasharray="4 4" />
                  <ReferenceLine y={-thresholdX} stroke="#60a5fa" strokeDasharray="4 4" />
                </>
              )}

              {chartMode === 'forecast' ? (
                <Line
                  type="monotone"
                  dataKey="netLoad"
                  stroke="#a78bfa"
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
                    stroke="#a78bfa"
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
              <span className="rounded-lg border border-slate-500/50 bg-slate-900/80 px-2.5 py-1 text-slate-200 shadow-sm">
                Positive net load = undersupply
              </span>
              <span className="rounded-lg border border-slate-500/50 bg-slate-900/80 px-2.5 py-1 text-slate-200 shadow-sm">
                Negative net load = oversupply
              </span>
            </>
          ) : (
            <span className="rounded-lg border border-slate-500/50 bg-slate-900/80 px-2.5 py-1 text-slate-200 shadow-sm">
              Comparison view is shown only when actual values are available
            </span>
          )}
        </div>
      </div>

      <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="card-header text-white">Action Detail View</h2>
            <p className="mt-1 text-sm text-slate-300">
              Switch between a quick priority summary and the full daily action timeline.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-xl border border-slate-600/60 bg-slate-900/75 p-1 shadow-sm">
            <button
              type="button"
              onClick={() => setViewMode('priority')}
              className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
                viewMode === 'priority'
                  ? 'border-violet-400/30 bg-violet-900/30 text-violet-100 shadow-sm'
                  : 'border-slate-600 bg-slate-800/80 text-slate-200 hover:bg-slate-700'
              }`}
            >
              Priority View
            </button>
            <button
              type="button"
              onClick={() => setViewMode('full')}
              className={`rounded-lg border px-3 py-1.5 text-sm font-medium transition ${
                viewMode === 'full'
                  ? 'border-violet-400/30 bg-violet-900/30 text-violet-100 shadow-sm'
                  : 'border-slate-600 bg-slate-800/80 text-slate-200 hover:bg-slate-700'
              }`}
            >
              Full Day View
            </button>
          </div>
        </div>
      </div>

      {viewMode === 'priority' ? (
        <>
          <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
            <div className="mb-4 flex items-center gap-2">
              <Battery className="h-5 w-5 text-emerald-400" />
              <h2 className="card-header text-white">Action Summary</h2>
            </div>

            {!actionSummary ? (
              <p className="text-sm text-slate-300">Run a forecast to view action priorities.</p>
            ) : (
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                <div className="rounded-2xl border border-red-400/35 bg-red-900/30 p-5 shadow-md">
                  <div className="mb-2 flex items-center gap-2">
                    <ArrowUpFromLine className="h-4 w-4 text-red-300" />
                    <p className="font-semibold text-red-100">Highest Undersupply Risk</p>
                  </div>
                  {actionSummary.worstUnder ? (
                    <>
                      <p className="text-lg font-semibold text-white">{fmtTime(actionSummary.worstUnder.timestamp)}</p>
                      <p className="mt-1 text-sm text-red-100/85">{actionSummary.worstUnder.yhat_mw.toFixed(2)} MW</p>
                      <span
                        className={`mt-3 inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${actionBadgeClass(
                          actionSummary.worstUnder.recommended_action
                        )}`}
                      >
                        {actionSummary.worstUnder.recommended_action}
                      </span>
                    </>
                  ) : (
                    <p className="text-sm text-slate-300">No major undersupply interval.</p>
                  )}
                </div>

                <div className="rounded-2xl border border-blue-400/35 bg-blue-900/30 p-5 shadow-md">
                  <div className="mb-2 flex items-center gap-2">
                    <ArrowDownToLine className="h-4 w-4 text-blue-300" />
                    <p className="font-semibold text-blue-100">Highest Oversupply Risk</p>
                  </div>
                  {actionSummary.worstOver ? (
                    <>
                      <p className="text-lg font-semibold text-white">{fmtTime(actionSummary.worstOver.timestamp)}</p>
                      <p className="mt-1 text-sm text-blue-100/85">{actionSummary.worstOver.yhat_mw.toFixed(2)} MW</p>
                      <span
                        className={`mt-3 inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${actionBadgeClass(
                          actionSummary.worstOver.recommended_action
                        )}`}
                      >
                        {actionSummary.worstOver.recommended_action}
                      </span>
                    </>
                  ) : (
                    <p className="text-sm text-slate-300">No major oversupply interval.</p>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="card-header text-white">Priority Actions</h2>
              <span className="text-xs font-medium text-slate-300">Top 5 intervals</span>
            </div>

            {!imbalance ? (
              <p className="text-sm text-slate-300">Run a forecast to generate actions.</p>
            ) : (
              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-600/60 bg-slate-900/55 p-4 shadow-sm">
                  <h3 className="mb-3 font-semibold text-white">Top Undersupply</h3>
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="text-slate-300">
                        <tr>
                          <th className="py-2 text-left font-medium">Time</th>
                          <th className="py-2 text-left font-medium">MW</th>
                          <th className="py-2 text-left font-medium">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topUndersupply.map((r, idx) => (
                          <tr key={idx} className="border-t border-slate-700/80">
                            <td className="py-3 text-slate-100">{fmtTime(r.timestamp)}</td>
                            <td className="py-3 font-medium text-slate-100">{r.yhat_mw.toFixed(2)}</td>
                            <td className="py-3">
                              <span
                                className={`inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${actionBadgeClass(
                                  r.recommended_action
                                )}`}
                              >
                                {r.recommended_action}
                              </span>
                            </td>
                          </tr>
                        ))}
                        {!topUndersupply.length && (
                          <tr>
                            <td colSpan={3} className="py-3 text-slate-300">
                              No undersupply intervals above threshold.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-600/60 bg-slate-900/55 p-4 shadow-sm">
                  <h3 className="mb-3 font-semibold text-white">Top Oversupply</h3>
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="text-slate-300">
                        <tr>
                          <th className="py-2 text-left font-medium">Time</th>
                          <th className="py-2 text-left font-medium">MW</th>
                          <th className="py-2 text-left font-medium">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topOversupply.map((r, idx) => (
                          <tr key={idx} className="border-t border-slate-700/80">
                            <td className="py-3 text-slate-100">{fmtTime(r.timestamp)}</td>
                            <td className="py-3 font-medium text-slate-100">{r.yhat_mw.toFixed(2)}</td>
                            <td className="py-3">
                              <span
                                className={`inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${actionBadgeClass(
                                  r.recommended_action
                                )}`}
                              >
                                {r.recommended_action}
                              </span>
                            </td>
                          </tr>
                        ))}
                        {!topOversupply.length && (
                          <tr>
                            <td colSpan={3} className="py-3 text-slate-300">
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

          <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
            <div className="mb-3 flex items-center gap-2">
              <Zap className="h-5 w-5 text-amber-400" />
              <h2 className="card-header text-white">Forecast Insight</h2>
            </div>

            {!insightText ? (
              <p className="text-sm text-slate-300">Run a forecast to generate operational insight.</p>
            ) : (
              <p className="leading-7 text-slate-200">{insightText}</p>
            )}
          </div>
        </>
      ) : (
        <div className="card border border-slate-600/70 bg-slate-800/85 shadow-xl">
          <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="card-header text-white">Full Day Action Timeline</h2>
              <p className="mt-1 text-sm text-slate-300">
                Showing the daily action profile at {resolutionLabel(fullDayResolution)} resolution.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-200">Resolution:</label>
                <select
                  value={fullDayResolution}
                  onChange={(e) => setFullDayResolution(e.target.value as Resolution)}
                  className="input-field border-slate-500 bg-slate-900/90 text-slate-100"
                >
                  <option value="15m">15 min</option>
                  <option value="30m">30 min</option>
                  <option value="1h">1 hour</option>
                  <option value="2h">2 hours</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-200">State:</label>
                <select
                  value={fullDayStateFilter}
                  onChange={(e) => setFullDayStateFilter(e.target.value as StateFilter)}
                  className="input-field border-slate-500 bg-slate-900/90 text-slate-100"
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
            <p className="text-sm text-slate-300">Run a forecast to view the full daily action timeline.</p>
          ) : (
            <div className="overflow-auto rounded-2xl border border-slate-600/60 bg-slate-900/55 p-2 shadow-sm">
              <table className="w-full text-sm">
                <thead className="text-slate-300">
                  <tr>
                    <th className="py-2 text-left font-medium">Time</th>
                    <th className="py-2 text-left font-medium">Forecast (MW)</th>
                    <th className="py-2 text-left font-medium">State</th>
                    <th className="py-2 text-left font-medium">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {fullDayRows.map((r, idx) => (
                    <tr key={`${r.step_idx}-${idx}`} className="border-t border-slate-700/80">
                      <td className="py-3 text-slate-100">{fmtTime(r.timestamp)}</td>
                      <td className="py-3 text-slate-100">{r.yhat_mw.toFixed(2)}</td>
                      <td className="py-3">
                        <span className={`inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${stateBadgeClass(r.label)}`}>
                          {r.label}
                        </span>
                      </td>
                      <td className="py-3">
                        <span
                          className={`inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${actionBadgeClass(
                            r.recommended_action
                          )}`}
                        >
                          {r.recommended_action}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {!fullDayRows.length && (
                    <tr>
                      <td colSpan={4} className="py-3 text-slate-300">
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