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
import { TrendingUp, Sun, Zap } from "lucide-react";
import api from "../services/api";
import { useGridStore } from "../stores/gridStore";
import type { SolarDayData } from "../types";

const INSTALLED_CAPACITY = 53.4;

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

  // Current actual solar MW for this hour
  const currentActualMw = currentDay ? currentDay.actual[currentHour] ?? null : null;

  // Current predicted solar MW for this hour (from PREVIOUS day's forecast)
  const prevDay = history.length > 1 ? history[history.length - 2] : null;
  const currentPredictedMw =
    prevDay && prevDay.predictedNextDay.length > 0
      ? prevDay.predictedNextDay[currentHour] ?? null
      : null;

  // Chart data: current day's actual vs prediction (from yesterday's forecast)
  const chartData = Array.from({ length: 24 }, (_, i) => {
    const label = `${String(i).padStart(2, "0")}:00`;
    const actual = currentDay ? currentDay.actual[i] : undefined;
    const predicted =
      prevDay && prevDay.predictedNextDay.length > 0
        ? prevDay.predictedNextDay[i]
        : undefined;
    return {
      hour: i,
      label,
      actual: actual !== undefined ? actual : null,
      predicted: predicted !== undefined ? predicted : null,
      capacity: INSTALLED_CAPACITY,
    };
  });

  // Analysis
  const peakActual = currentDay
    ? Math.max(...currentDay.actual, 0)
    : 0;
  const excessHours = currentDay
    ? currentDay.actual.filter((v) => v > INSTALLED_CAPACITY).length
    : 0;

  // Recommendation
  const maxExcess = currentDay
    ? Math.max(...currentDay.actual.map((v) => v - INSTALLED_CAPACITY), 0)
    : 0;
  let recommendation = "Grid operating within safe limits.";
  if (maxExcess > 20) {
    recommendation = "Activate battery storage immediately and curtail solar output.";
  } else if (maxExcess > 10) {
    recommendation = "Reduce solar generation and consider battery charging.";
  } else if (maxExcess > 5) {
    recommendation = "Minor curtailment or load shifting recommended.";
  }

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
            <p className="text-slate-400 text-sm">Current Actual Solar</p>
          </div>
          <p className="text-2xl font-bold text-green-400">
            {currentActualMw !== null ? `${currentActualMw.toFixed(2)} MW` : "--"}
          </p>
          {isActive && (
            <p className="text-xs text-slate-500 mt-1">
              {String(currentHour).padStart(2, "0")}:00
            </p>
          )}
        </div>

        <div className="card text-center">
          <div className="flex items-center justify-center mb-1">
            <Zap className="w-4 h-4 text-blue-400 mr-1" />
            <p className="text-slate-400 text-sm">Current Predicted Solar</p>
          </div>
          <p className="text-2xl font-bold text-blue-400">
            {currentPredictedMw !== null
              ? `${currentPredictedMw.toFixed(2)} MW`
              : "--"}
          </p>
          {currentPredictedMw === null && isActive && (
            <p className="text-xs text-slate-500 mt-1">
              Available from Day 2
            </p>
          )}
        </div>

        <div className="card text-center">
          <p className="text-slate-400 text-sm">Peak Actual (Today)</p>
          <p className="text-2xl font-bold text-yellow-400">
            {peakActual > 0 ? `${peakActual.toFixed(2)} MW` : "--"}
          </p>
        </div>

        <div className="card text-center">
          <p className="text-slate-400 text-sm">Risk Hours (Today)</p>
          <p className="text-2xl font-bold text-red-400">
            {currentDay ? excessHours : "--"}
          </p>
        </div>
      </div>

      {/* Chart: Today's Actual vs Yesterday's Prediction */}
      {currentDay && (
        <div className="card">
          <h2 className="card-header">
            {currentDay.date} — Actual vs Predicted Solar Generation
          </h2>
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
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#22c55e"
                  strokeWidth={3}
                  dot={false}
                  name="Actual Solar (MW)"
                  connectNulls
                />
                {prevDay && prevDay.predictedNextDay.length > 0 && (
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    name="Predicted Solar (MW)"
                    connectNulls
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="capacity"
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  strokeWidth={1.5}
                  dot={false}
                  name="Installed Capacity (MW)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Recommendation */}
      {currentDay && (
        <div className="card bg-slate-800 border border-purple-500">
          <h2 className="card-header">Smart Control Recommendation</h2>
          <p className="text-sm text-slate-300 mt-3">{recommendation}</p>
        </div>
      )}

      {/* History Table */}
      {history.length > 0 && (
        <div className="card">
          <h2 className="card-header">
            Solar Generation History (Actual vs Predicted)
          </h2>
          <div className="overflow-x-auto mt-3">
            <table className="text-sm text-slate-300 border-collapse">
              <thead>
                <tr className="border-b border-slate-600">
                  <th className="text-left py-2 px-3 sticky left-0 bg-slate-800 z-10 min-w-[70px]">
                    Hour
                  </th>
                  {history.map((entry, idx) => (
                    <th
                      key={`actual-${idx}`}
                      colSpan={1}
                      className="text-center py-2 px-3 min-w-[120px] text-green-400"
                    >
                      {entry.date}
                      <br />
                      <span className="text-xs font-normal">Actual</span>
                    </th>
                  ))}
                  {history.map((entry, idx) =>
                    entry.predictedNextDay.length > 0 ? (
                      <th
                        key={`pred-${idx}`}
                        colSpan={1}
                        className="text-center py-2 px-3 min-w-[120px] text-blue-400"
                      >
                        {entry.nextDate}
                        <br />
                        <span className="text-xs font-normal">Predicted</span>
                      </th>
                    ) : null
                  )}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 24 }, (_, hr) => (
                  <tr
                    key={hr}
                    className={`border-b border-slate-700/50 ${
                      isActive && hr === currentHour ? "bg-slate-700/40" : ""
                    }`}
                  >
                    <td className="py-1 px-3 sticky left-0 bg-slate-800 z-10 font-mono text-xs">
                      {String(hr).padStart(2, "0")}:00
                    </td>
                    {history.map((entry, idx) => (
                      <td
                        key={`a-${idx}`}
                        className="text-right py-1 px-3 font-mono text-xs"
                      >
                        {entry.actual[hr]?.toFixed(2) ?? "--"}
                      </td>
                    ))}
                    {history.map((entry, idx) =>
                      entry.predictedNextDay.length > 0 ? (
                        <td
                          key={`p-${idx}`}
                          className="text-right py-1 px-3 font-mono text-xs"
                        >
                          {entry.predictedNextDay[hr]?.toFixed(2) ?? "--"}
                        </td>
                      ) : null
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

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
