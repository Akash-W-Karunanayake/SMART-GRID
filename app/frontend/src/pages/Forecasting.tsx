import { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingUp, Sun, Zap, Settings } from "lucide-react";
import api from "../services/api";
import { useGridStore } from "../stores/gridStore";
import type { SolarDayData } from "../types";

interface DayEntry {
  date: string;
  nextDate: string;
  actual: number[];
  predictedNextDay: number[];
}

export default function Forecasting() {
  const {
    playbackPlaying,
    playbackCurrentDate,
    playbackDayIndex,
    playbackTotalDays,
    liveMetrics,
  } = useGridStore();

  const [history, setHistory] = useState<DayEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fetchedDatesRef = useRef<Set<string>>(new Set());
  const [installedCapacity, setInstalledCapacity] = useState(53.4);

  // Line visibility toggles
  const [showActual, setShowActual] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);
  const [showCapacity, setShowCapacity] = useState(true);
  const [showNextDay, setShowNextDay] = useState(true);

  const isActive = playbackPlaying;

  // Reset history when a new simulation starts (dayIndex resets to 0)
  const prevPlayingRef = useRef(false);
  useEffect(() => {
    if (isActive && !prevPlayingRef.current) {
      setHistory([]);
      fetchedDatesRef.current.clear();
      setError(null);
    }
    prevPlayingRef.current = isActive;
  }, [isActive]);

  // Fetch solar day data when a new day begins
  useEffect(() => {
    if (!playbackCurrentDate) return;
    if (fetchedDatesRef.current.has(playbackCurrentDate)) return;

    fetchedDatesRef.current.add(playbackCurrentDate);
    setLoading(true);

    api
      .getSolarDayData(playbackCurrentDate)
      .then((data: unknown) => {
        const d = data as SolarDayData;
        setHistory((prev) => [
          ...prev,
          {
            date: d.date,
            nextDate: d.next_date,
            actual: d.actual_mw,
            predictedNextDay: d.predicted_next_day_mw,
          },
        ]);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : "Failed to fetch solar data";
        setError(msg);
      })
      .finally(() => setLoading(false));
  }, [playbackCurrentDate]);

  // Current day entry
  const currentDay = history.length > 0 ? history[history.length - 1] : null;

  // Current hour (from 15-min step index: step 0-3 = hour 0, 4-7 = hour 1, etc.)
  const currentHour = liveMetrics ? Math.floor(liveMetrics.hour) : 0;

  // Previous day entry (for yesterday's forecast of today)
  const prevDay = history.length > 1 ? history[history.length - 2] : null;

  // Next-day spotlight: predicted solar for current hour tomorrow
  const nextDayCurrentHourMw =
    currentDay && currentDay.predictedNextDay.length > 0
      ? currentDay.predictedNextDay[currentHour] ?? null
      : null;
  const todayActualCurrentHourMw =
    currentDay ? currentDay.actual[currentHour] ?? null : null;
  const nextDayDelta =
    nextDayCurrentHourMw !== null && todayActualCurrentHourMw !== null
      ? nextDayCurrentHourMw - todayActualCurrentHourMw
      : null;

  // Chart data: current day's actual vs prediction (from yesterday's forecast) + next-day forecast
  const chartData = Array.from({ length: 24 }, (_, i) => {
    const label = `${String(i).padStart(2, "0")}:00`;
    const actual = currentDay ? currentDay.actual[i] : undefined;
    const predicted =
      prevDay && prevDay.predictedNextDay.length > 0
        ? prevDay.predictedNextDay[i]
        : undefined;
    const nextDay =
      currentDay && currentDay.predictedNextDay.length > 0
        ? currentDay.predictedNextDay[i]
        : undefined;
    return {
      hour: i,
      label,
      actual: showActual ? (actual !== undefined ? actual : null) : null,
      predicted: showPredicted ? (predicted !== undefined ? predicted : null) : null,
      capacity: showCapacity ? installedCapacity : null,
      nextDay: showNextDay ? (nextDay !== undefined ? nextDay : null) : null,
    };
  });

  // Analysis — use predicted peak (from current day's next-day forecast)
  const peakPredicted = currentDay
    ? Math.max(...currentDay.predictedNextDay, 0)
    : 0;

  // Solar Output Level = peak predicted / installed capacity
  const maxSolarOutputLevel = installedCapacity > 0 && peakPredicted > 0
    ? peakPredicted / installedCapacity
    : 0;

  // Per-hour solar output level and recommendation lookup (per Reco.txt rules)
  const getSolarOutputInfo = (level: number) => {
    if (level >= 0.90) return { risk: "Reverse power flow, voltage rise", action: "Charge batteries, reduce solar output if needed", color: "text-red-400", bg: "bg-red-900/20" };
    if (level >= 0.75) return { risk: "Possible power flow congestion", action: "Monitor grid conditions, use flexible loads", color: "text-amber-400", bg: "bg-amber-900/20" };
    if (level >= 0.50) return { risk: "Grid operating normally", action: "Continue normal operation", color: "text-green-400", bg: "" };
    if (level >= 0.01) return { risk: "Reduced renewable supply", action: "Prepare backup generation", color: "text-blue-400", bg: "" };
    return { risk: "No solar generation", action: "Start backup generators or discharge batteries", color: "text-purple-400", bg: "" };
  };

  // Expand/collapse state for recommendation table
  const [recoExpanded, setRecoExpanded] = useState(false);

  // Has predicted data available
  const hasPredicted = prevDay && prevDay.predictedNextDay.length > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
            Solar Generation Forecast
          </h1>
          <p className="text-slate-400 text-sm">
            {isActive ? (
              <span className="text-green-400">
                Live — Simulation Running
                {playbackCurrentDate && ` — ${playbackCurrentDate}`}
                {playbackTotalDays > 1 &&
                  ` (Day ${playbackDayIndex + 1}/${playbackTotalDays})`}
              </span>
            ) : history.length > 0 ? (
              <span className="text-slate-400">
                Simulation Complete — {history.length} day(s) processed
              </span>
            ) : (
              <span className="text-slate-500">
                Idle — Start simulation to begin forecasting
              </span>
            )}
          </p>
        </div>
        {loading && (
          <span className="text-xs text-blue-400 animate-pulse">
            Loading solar data...
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-500 rounded-lg p-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Live Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <div className="flex items-center justify-center mb-1">
            <Sun className="w-4 h-4 text-yellow-400 mr-1" />
            <p className="text-slate-400 text-sm">Peak Predicted Solar (24h)</p>
          </div>
          <p className="text-2xl font-bold text-yellow-400">
            {peakPredicted > 0 ? `${peakPredicted.toFixed(2)} MW` : "--"}
          </p>
          {currentDay && (
            <p className="text-xs text-slate-500 mt-1">
              Next day: {currentDay.nextDate}
            </p>
          )}
        </div>

        {/* Next-Day Spotlight Card */}
        <div className={`card text-center border ${nextDayCurrentHourMw !== null && nextDayCurrentHourMw > installedCapacity
            ? "border-red-500/60 bg-red-900/10"
            : nextDayCurrentHourMw !== null && nextDayCurrentHourMw > installedCapacity * 0.8
              ? "border-amber-500/60 bg-amber-900/10"
              : "border-slate-700"
          }`}>
          <div className="flex items-center justify-center mb-1">
            <TrendingUp className="w-4 h-4 text-orange-400 mr-1" />
            <p className="text-slate-400 text-sm">
              Tomorrow @ {String(currentHour).padStart(2, "0")}:00
            </p>
          </div>
          <p className={`text-2xl font-bold ${nextDayCurrentHourMw !== null && nextDayCurrentHourMw > installedCapacity
              ? "text-red-400"
              : "text-orange-400"
            }`}>
            {nextDayCurrentHourMw !== null
              ? `${nextDayCurrentHourMw.toFixed(2)} MW`
              : "--"}
          </p>
          {nextDayDelta !== null && (
            <p className={`text-xs mt-1 ${nextDayDelta >= 0 ? "text-amber-400" : "text-cyan-400"}`}>
              {nextDayDelta >= 0 ? "+" : ""}{nextDayDelta.toFixed(2)} MW vs today
            </p>
          )}
          {currentDay && (
            <p className="text-xs text-slate-600 mt-0.5">
              {currentDay.nextDate}
            </p>
          )}
        </div>

        <div className={`card text-center border ${
          maxSolarOutputLevel >= 0.90
            ? "border-red-500/60 bg-red-900/10"
            : maxSolarOutputLevel >= 0.75
            ? "border-amber-500/60 bg-amber-900/10"
            : "border-slate-700"
        }`}>
          <div className="flex items-center justify-center mb-1">
            <Zap className="w-4 h-4 mr-1" style={{ color: maxSolarOutputLevel >= 0.90 ? '#f87171' : maxSolarOutputLevel >= 0.75 ? '#fbbf24' : '#94a3b8' }} />
            <p className="text-slate-400 text-sm">Max Solar Output Level</p>
          </div>
          <p className={`text-2xl font-bold ${getSolarOutputInfo(maxSolarOutputLevel).color}`}>
            {currentDay ? maxSolarOutputLevel.toFixed(2) : "--"}
          </p>
          {currentDay && (
            <p className={`text-xs mt-1 ${getSolarOutputInfo(maxSolarOutputLevel).color}`}>
              {getSolarOutputInfo(maxSolarOutputLevel).risk}
            </p>
          )}
        </div>

        <div className="card text-center">
          <div className="flex items-center justify-center mb-1">
            <Settings className="w-4 h-4 text-slate-400 mr-1" />
            <p className="text-slate-400 text-sm">Installed Capacity</p>
          </div>
          <div className="flex items-center justify-center gap-1 mt-1">
            <input
              type="number"
              value={installedCapacity}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val) && val >= 0) setInstalledCapacity(val);
              }}
              className="bg-slate-700 border border-slate-600 text-white text-center text-lg font-bold rounded px-2 py-1 w-24 focus:outline-none focus:border-blue-500"
              step="0.1"
              min="0"
            />
            <span className="text-sm text-slate-400">MW</span>
          </div>
        </div>
      </div>

      {/* Chart: Today's Actual vs Yesterday's Prediction + optional Load */}
      {currentDay && (
        <div className="card">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <h2 className="card-header mb-0">
              {currentDay.date} — Actual vs Predicted Solar Generation
            </h2>
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showActual}
                  onChange={(e) => setShowActual(e.target.checked)}
                  className="w-3.5 h-3.5 rounded cursor-pointer accent-green-500"
                />
                <span className="text-green-400">Actual</span>
              </label>
              <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showPredicted}
                  onChange={(e) => setShowPredicted(e.target.checked)}
                  className="w-3.5 h-3.5 rounded cursor-pointer accent-blue-500"
                />
                <span className="text-blue-400">Predicted</span>
              </label>
              <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showNextDay}
                  onChange={(e) => setShowNextDay(e.target.checked)}
                  className="w-3.5 h-3.5 rounded cursor-pointer accent-orange-500"
                />
                <span className="text-orange-400">Next Day</span>
              </label>
              <label className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showCapacity}
                  onChange={(e) => setShowCapacity(e.target.checked)}
                  className="w-3.5 h-3.5 rounded cursor-pointer accent-red-500"
                />
                <span className="text-red-400">Capacity</span>
              </label>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="label" stroke="#9ca3af" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "1px solid #475569",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Legend />
                {showActual && (
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#22c55e"
                    strokeWidth={3}
                    dot={false}
                    name={`Actual Solar (${currentDay?.date})`}
                    connectNulls
                  />
                )}
                {showPredicted && hasPredicted && (
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    name={`Predicted Solar (${currentDay?.date})`}
                    connectNulls
                  />
                )}
                {showNextDay && currentDay && currentDay.predictedNextDay.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="nextDay"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    dot={false}
                    name={`Next Day Forecast (${currentDay.nextDate})`}
                    connectNulls
                  />
                )}
                {showCapacity && (
                  <Line
                    type="monotone"
                    dataKey="capacity"
                    stroke="#ef4444"
                    strokeDasharray="5 5"
                    strokeWidth={1.5}
                    dot={false}
                    name="Installed Capacity (MW)"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Smart Control Recommendation Table — rows appear progressively as each hour arrives */}
      {currentDay && currentDay.predictedNextDay.length > 0 && (() => {
        // Only show rows up to the current simulation hour (progressive reveal)
        const visibleRows = isActive
          ? Array.from({ length: currentHour + 1 }, (_, i) => i)
          : Array.from({ length: 24 }, (_, i) => i);
        const defaultVisible = 5;
        const displayRows = recoExpanded ? visibleRows : visibleRows.slice(0, defaultVisible);
        const hasMore = visibleRows.length > defaultVisible;

        return (
          <div className="card">
            <div className="flex items-center justify-between mb-0">
              <h2 className="card-header">
                Smart Control Recommendation — {currentDay.nextDate}
              </h2>
              <span className="text-xs text-slate-500">
                {visibleRows.length} / 24 hours
              </span>
            </div>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm text-slate-300 border-collapse">
                <thead>
                  <tr className="border-b border-slate-600">
                    <th className="text-left py-2 px-3 min-w-[70px]">Hour</th>
                    <th className="text-right py-2 px-3 min-w-[110px]">Predicted Solar</th>
                    <th className="text-right py-2 px-3 min-w-[100px]">Capacity</th>
                    <th className="text-right py-2 px-3 min-w-[120px]">Solar Output Level</th>
                    <th className="text-left py-2 px-3 min-w-[180px]">Risk</th>
                    <th className="text-left py-2 px-3 min-w-[250px]">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {displayRows.map((hr) => {
                    const predictedMw = currentDay.predictedNextDay[hr] ?? 0;
                    const level = installedCapacity > 0 ? predictedMw / installedCapacity : 0;
                    const info = getSolarOutputInfo(level);
                    const isLatest = isActive && hr === currentHour;
                    return (
                      <tr
                        key={hr}
                        className={`border-b border-slate-700/50 ${info.bg} ${
                          isLatest ? "ring-1 ring-inset ring-slate-500" : ""
                        }`}
                      >
                        <td className="py-1.5 px-3 font-mono text-xs">
                          {String(hr).padStart(2, "0")}:00
                          {isLatest && (
                            <span className="ml-1.5 text-[10px] text-green-400 animate-pulse">LIVE</span>
                          )}
                        </td>
                        <td className="text-right py-1.5 px-3 font-mono text-xs">
                          {predictedMw.toFixed(2)} MW
                        </td>
                        <td className="text-right py-1.5 px-3 font-mono text-xs text-slate-500">
                          {installedCapacity.toFixed(1)} MW
                        </td>
                        <td className={`text-right py-1.5 px-3 font-mono text-xs font-semibold ${info.color}`}>
                          {level.toFixed(2)}
                        </td>
                        <td className={`py-1.5 px-3 text-xs ${info.color}`}>
                          {info.risk}
                        </td>
                        <td className="py-1.5 px-3 text-xs text-slate-400">
                          {info.action}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {hasMore && (
              <button
                onClick={() => setRecoExpanded((prev) => !prev)}
                className="mt-3 w-full py-1.5 text-xs text-slate-400 hover:text-white bg-slate-700/40 hover:bg-slate-700 rounded transition-colors"
              >
                {recoExpanded
                  ? "Show less"
                  : `Show all ${visibleRows.length} hours`}
              </button>
            )}
          </div>
        );
      })()}

      {/* Empty state */}
      {!isActive && history.length === 0 && (
        <div className="card text-center py-12">
          <Sun className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">
            Select a date range and click <strong>Run</strong> to start the
            simulation. Solar forecasting data will appear here automatically.
          </p>
        </div>
      )}
    </div>
  );
}
