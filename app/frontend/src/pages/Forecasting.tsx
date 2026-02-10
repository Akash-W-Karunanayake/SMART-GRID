import { useState, useEffect } from 'react';
import { TrendingUp, Sun, Home, RefreshCw, Bell } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Legend,
} from 'recharts';
import api from '../services/api';
import type { ForecastResponse } from '../types';

export default function Forecasting() {
  const [solarForecast, setSolarForecast] = useState<ForecastResponse | null>(null);
  const [loadForecast, setLoadForecast] = useState<ForecastResponse | null>(null);
  const [imbalance, setImbalance] = useState<any>(null);
  const [alerts, setAlerts] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [horizonHours, setHorizonHours] = useState(24);

  const loadForecasts = async () => {
    setLoading(true);
    try {
      const [solar, load, imb, alertData] = await Promise.all([
        api.forecastSolar(horizonHours),
        api.forecastLoad(horizonHours),
        api.detectImbalance(),
        api.getHouseholdAlerts(),
      ]);
      setSolarForecast(solar);
      setLoadForecast(load);
      setImbalance(imb);
      setAlerts(alertData);
    } catch (error) {
      console.error('Failed to load forecasts:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadForecasts();
  }, [horizonHours]);

  // Combine forecasts for chart
  const chartData = solarForecast?.points.map((point, i) => ({
    hour: i,
    solar: point.value,
    solarLower: point.lower_bound,
    solarUpper: point.upper_bound,
    load: loadForecast?.points[i]?.value || 0,
    loadLower: loadForecast?.points[i]?.lower_bound || 0,
    loadUpper: loadForecast?.points[i]?.upper_bound || 0,
  })) || [];

  const imbalanceColors = {
    balanced: 'text-green-400 bg-green-900/30',
    oversupply: 'text-blue-400 bg-blue-900/30',
    undersupply: 'text-red-400 bg-red-900/30',
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
            Solar Forecasting & Grid Stability
          </h1>
          <p className="text-slate-400">
            Stacked Ensemble ML model for weather-based solar prediction and household consumption analysis
          </p>
        </div>
        <button
          onClick={loadForecasts}
          disabled={loading}
          className="btn-secondary flex items-center"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Research Info Banner */}
      <div className="card bg-green-900/30 border-green-700">
        <div className="flex items-start">
          <Sun className="w-6 h-6 text-green-400 flex-shrink-0" />
          <div className="ml-4">
            <h3 className="text-white font-medium">Component 2: IT22360182 - Daraniyagala G.K</h3>
            <p className="text-slate-300 text-sm mt-1">
              Intelligent system for predicting grid stability issues using solar forecasting and household
              solar consumption analysis. Uses stacked ensemble model (RF + LSTM + GBR meta-learner).
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-green-800 text-green-200 text-xs rounded">Random Forest</span>
              <span className="px-2 py-1 bg-green-800 text-green-200 text-xs rounded">LSTM</span>
              <span className="px-2 py-1 bg-green-800 text-green-200 text-xs rounded">GRU</span>
              <span className="px-2 py-1 bg-green-800 text-green-200 text-xs rounded">Gradient Boosting</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center space-x-4">
        <label className="text-slate-400 text-sm">Forecast Horizon:</label>
        <select
          value={horizonHours}
          onChange={(e) => setHorizonHours(Number(e.target.value))}
          className="input-field"
        >
          <option value={12}>12 hours</option>
          <option value={24}>24 hours</option>
          <option value={48}>48 hours</option>
          <option value={72}>72 hours</option>
        </select>
      </div>

      {/* Main Chart */}
      <div className="card">
        <h2 className="card-header">Solar Generation vs Load Forecast</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" label={{ value: 'Hours', position: 'bottom' }} />
              <YAxis stroke="#9ca3af" label={{ value: 'kW', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="solarUpper"
                stroke="transparent"
                fill="#22c55e"
                fillOpacity={0.1}
                name="Solar Upper"
              />
              <Area
                type="monotone"
                dataKey="solarLower"
                stroke="transparent"
                fill="#22c55e"
                fillOpacity={0.1}
                name="Solar Lower"
              />
              <Line
                type="monotone"
                dataKey="solar"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                name="Solar Generation"
              />
              <Line
                type="monotone"
                dataKey="load"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Load Demand"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Current Imbalance Status */}
        <div className="card">
          <h2 className="card-header">Imbalance Detection</h2>
          {imbalance && (
            <div className={`p-4 rounded-lg ${imbalanceColors[imbalance.current_state as keyof typeof imbalanceColors]}`}>
              <p className="text-lg font-semibold capitalize">{imbalance.current_state}</p>
              <p className="text-sm mt-1">Net Load: {imbalance.net_load_kw} kW</p>
              <p className="text-sm">Confidence: {(imbalance.confidence * 100).toFixed(0)}%</p>
            </div>
          )}
          <div className="mt-4 p-3 bg-slate-700 rounded-lg">
            <p className="text-xs text-slate-400">{imbalance?.recommendation}</p>
          </div>
        </div>

        {/* Household Alerts */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Bell className="w-5 h-5 mr-2 text-amber-400" />
            Household Alerts
          </h2>
          <div className="space-y-2">
            {alerts?.alerts?.map((alert: any, i: number) => (
              <div
                key={i}
                className={`p-3 rounded-lg ${
                  alert.type === 'warning' ? 'bg-amber-900/30 border-l-2 border-amber-500' :
                  'bg-blue-900/30 border-l-2 border-blue-500'
                }`}
              >
                <p className="text-sm text-white">{alert.message}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Home className="w-5 h-5 mr-2 text-purple-400" />
            Recommendations
          </h2>
          <ul className="space-y-2">
            {alerts?.recommendations?.map((rec: string, i: number) => (
              <li key={i} className="text-sm text-slate-300 flex items-start">
                <span className="text-green-400 mr-2">•</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Model Info */}
      <div className="card bg-slate-700/50">
        <p className="text-slate-400 text-sm">
          <span className="text-amber-400 font-medium">Note:</span> Currently displaying mock forecast data.
          The stacked ensemble model (Random Forest + LSTM base learners with Gradient Boosting meta-learner)
          will be integrated when the ML training is complete.
        </p>
      </div>
    </div>
  );
}
