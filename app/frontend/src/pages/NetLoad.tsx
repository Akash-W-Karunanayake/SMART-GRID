import { useState, useEffect } from 'react';
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
} from 'recharts';
import api from '../services/api';

type NetLoadPoint = { timestamp: string; yhat_mw: number };
type NetLoadForecastResponse = { target_date: string; issue_time: string; points: NetLoadPoint[] };

export default function NetLoad() {
  const [netLoadForecast, setNetLoadForecast] = useState<NetLoadForecastResponse | null>(null);
  const [operatorData, setOperatorData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // default can be any valid date; keep one you tested in Swagger
  const [targetDate, setTargetDate] = useState('2025-08-10');

  const loadData = async () => {
    setLoading(true);
    try {
      const [netLoad, opData] = await Promise.all([
        api.forecastNetLoadByDate(targetDate) as Promise<NetLoadForecastResponse>,
        api.getGridOperatorData(),
      ]);
      setNetLoadForecast(netLoad);
      setOperatorData(opData);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetDate]);

  const chartData =
    netLoadForecast?.points.map((point, i) => ({
      hour: i,
      netLoad: point.yhat_mw,
      state: point.yhat_mw > 100 ? 'undersupply' : point.yhat_mw < -50 ? 'oversupply' : 'balanced',
    })) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-purple-400" />
            Net Load Forecasting
          </h1>
          <p className="text-slate-400">
            Probabilistic net load forecasting with power flow validation for renewable-based microgrids
          </p>
        </div>
        <button onClick={loadData} disabled={loading} className="btn-secondary flex items-center">
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Research Info Banner */}
      <div className="card bg-purple-900/30 border-purple-700">
        <div className="flex items-start">
          <BarChart3 className="w-6 h-6 text-purple-400 flex-shrink-0" />
          <div className="ml-4">
            <h3 className="text-white font-medium">Component 4: IT22891204 - Wickramaratne A J S de Z</h3>
            <p className="text-slate-300 text-sm mt-1">
              End-to-end framework that couples advanced hybrid net load forecasting with schedule-aware
              grid validation. Uses ICEEMDAN for signal decomposition, Transformer for temporal predictions,
              and Gaussian Process (GP-RML) for bias correction and uncertainty quantification.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-purple-800 text-purple-200 text-xs rounded">ICEEMDAN</span>
              <span className="px-2 py-1 bg-purple-800 text-purple-200 text-xs rounded">Transformer</span>
              <span className="px-2 py-1 bg-purple-800 text-purple-200 text-xs rounded">GP-RML</span>
              <span className="px-2 py-1 bg-purple-800 text-purple-200 text-xs rounded">Power Flow</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center space-x-4">
        <label className="text-slate-400 text-sm">Target Date:</label>
        <input
          type="date"
          value={targetDate}
          onChange={(e) => setTargetDate(e.target.value)}
          className="input-field"
          min="2025-07-15"
          max="2025-08-31"
        />
      </div>

      {/* Main Chart */}
      <div className="card">
        <h2 className="card-header">Net Load Forecast (Day-ahead)</h2>
        <p className="text-slate-400 text-sm mb-4">
          Net Load = Load Demand - Renewable Generation (positive = undersupply, negative = oversupply)
        </p>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" stroke="#9ca3af" label={{ value: '15-min Steps (0..95)', position: 'bottom' }} />
              <YAxis stroke="#9ca3af" label={{ value: 'MW', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                }}
                formatter={(value: any, name: string) => [`${Number(value).toFixed(3)} MW`, name]}
              />
              <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="3 3" label="Undersupply Threshold" />
              <ReferenceLine y={-50} stroke="#3b82f6" strokeDasharray="3 3" label="Oversupply Threshold" />
              <Line type="monotone" dataKey="netLoad" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Net Load" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="flex justify-center space-x-6 mt-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded mr-2" />
            <span className="text-slate-400">Net Load (Forecast)</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Next Hour Summary */}
        <div className="card">
          <h2 className="card-header">Next Hour Forecast</h2>
          {operatorData?.forecast_summary && (
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Load</span>
                <span className="text-white font-semibold">
                  {operatorData.forecast_summary.next_hour_load_kw} kW
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-700/50 rounded-lg">
                <span className="text-slate-400">Solar</span>
                <span className="text-green-400 font-semibold">
                  {operatorData.forecast_summary.next_hour_solar_kw} kW
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-purple-700/50 rounded-lg">
                <span className="text-slate-400">Net Load</span>
                <span className="text-purple-400 font-semibold">
                  {operatorData.forecast_summary.next_hour_net_load_kw} kW
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Storage Recommendation */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <Battery className="w-5 h-5 mr-2 text-green-400" />
            Battery Scheduling
          </h2>
          {operatorData?.storage_recommendation && (
            <div
              className={`p-4 rounded-lg ${
                operatorData.storage_recommendation.action === 'charge'
                  ? 'bg-blue-900/30 border border-blue-700'
                  : operatorData.storage_recommendation.action === 'discharge'
                  ? 'bg-amber-900/30 border border-amber-700'
                  : 'bg-slate-700/50'
              }`}
            >
              <p className="text-lg font-semibold text-white capitalize">
                {operatorData.storage_recommendation.action}
              </p>
              <p className="text-sm text-slate-300 mt-1">
                Target SoC: {operatorData.storage_recommendation.target_soc_percent}%
              </p>
              <p className="text-xs text-slate-400 mt-2">{operatorData.storage_recommendation.reason}</p>
            </div>
          )}
        </div>

        {/* Operator Alerts */}
        <div className="card">
          <h2 className="card-header flex items-center">
            <AlertCircle className="w-5 h-5 mr-2 text-amber-400" />
            Grid Operator Alerts
          </h2>
          <div className="space-y-2">
            {operatorData?.alerts?.map((alert: any, i: number) => (
              <div
                key={i}
                className={`p-3 rounded-lg ${
                  alert.severity === 'high'
                    ? 'bg-red-900/30 border-l-2 border-red-500'
                    : alert.severity === 'medium'
                    ? 'bg-amber-900/30 border-l-2 border-amber-500'
                    : 'bg-blue-900/30 border-l-2 border-blue-500'
                }`}
              >
                <p className="text-sm text-white">{alert.message}</p>
                <p className="text-xs text-slate-400 mt-1">{alert.action}</p>
              </div>
            ))}
            {!operatorData?.alerts?.length && (
              <div className="flex items-center text-green-400 p-3">
                <CheckCircle className="w-5 h-5 mr-2" />
                No alerts
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Imbalance States Legend */}
      <div className="card">
        <h2 className="card-header">Imbalance Response Strategy</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-red-900/20 rounded-lg border border-red-800">
            <h3 className="font-semibold text-red-400 mb-2">Undersupply</h3>
            <p className="text-sm text-slate-300">Net Load {'>'} 100 MW</p>
            <ul className="text-xs text-slate-400 mt-2 space-y-1">
              <li>• Discharge battery storage</li>
              <li>• Activate backup generation</li>
              <li>• Grid import if needed</li>
            </ul>
          </div>
          <div className="p-4 bg-green-900/20 rounded-lg border border-green-800">
            <h3 className="font-semibold text-green-400 mb-2">Balanced</h3>
            <p className="text-sm text-slate-300">-50 MW {'<'} Net Load {'<'} 100 MW</p>
            <ul className="text-xs text-slate-400 mt-2 space-y-1">
              <li>• Normal operation</li>
              <li>• Monitor for changes</li>
              <li>• Optimize efficiency</li>
            </ul>
          </div>
          <div className="p-4 bg-blue-900/20 rounded-lg border border-blue-800">
            <h3 className="font-semibold text-blue-400 mb-2">Oversupply</h3>
            <p className="text-sm text-slate-300">Net Load {'<'} -50 MW</p>
            <ul className="text-xs text-slate-400 mt-2 space-y-1">
              <li>• Charge battery storage</li>
              <li>• Curtail renewables if needed</li>
              <li>• Grid export if available</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="card bg-slate-700/50">
        <p className="text-slate-400 text-sm">
          <span className="text-amber-400 font-medium">Note:</span> Net load curve is now served by the backend
          Transformer model. GP-RML uncertainty bands will be added later.
        </p>
      </div>
    </div>
  );
}