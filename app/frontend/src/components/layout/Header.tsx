import { useState } from 'react';
import {
  Play,
  Pause,
  Square,
  SkipForward,
  Activity,
  Loader2,
  Calendar,
  Gauge,
} from 'lucide-react';
import { useGridStore } from '../../stores/gridStore';
import { useSimulationPlayback } from '../../hooks/useSimulationPlayback';

const SPEED_OPTIONS = [
  { label: '0.1s', ms: 100 },
  { label: '0.25s', ms: 250 },
  { label: '0.4s', ms: 400 },
  { label: '1s', ms: 1000 },
  { label: '2s', ms: 2000 },
];

export default function Header() {
  const {
    gridState,
    playbackPlaying,
    playbackPaused,
    playbackFetching,
    playbackCurrentDate,
    playbackStepIndex,
    playbackDayIndex,
    playbackTotalDays,
    playbackSpeed,
    liveMetrics,
    error,
  } = useGridStore();

  const playback = useSimulationPlayback();

  const [startDate, setStartDate] = useState('2025-08-01');
  const [endDate, setEndDate] = useState('2025-08-01');

  const isActive = playbackPlaying; // playback in progress (playing or paused)

  const handleRun = () => {
    playback.run(startDate, endDate);
  };

  const handlePause = () => {
    if (playbackPaused) {
      playback.resume();
    } else {
      playback.pause();
    }
  };

  const handleStop = () => {
    playback.stop();
  };

  const handleSkip = () => {
    playback.skip();
  };

  const handleSpeedChange = (ms: number) => {
    playback.setSpeed(ms);
  };

  // Format current hour for display
  const currentHourLabel = liveMetrics
    ? `${String(Math.floor(liveMetrics.hour)).padStart(2, '0')}:${String(Math.round((liveMetrics.hour % 1) * 60)).padStart(2, '0')}`
    : null;

  // Compute day count for display
  const dayCount = (() => {
    const s = new Date(startDate + 'T00:00:00');
    const e = new Date(endDate + 'T00:00:00');
    const diff = Math.floor((e.getTime() - s.getTime()) / 86400000) + 1;
    return diff > 0 ? diff : 1;
  })();

  return (
    <header className="flex items-center justify-between h-16 px-6 bg-slate-800 border-b border-slate-700">
      {/* Left: Date selection + control buttons */}
      <div className="flex items-center space-x-2">
        <Calendar className="w-4 h-4 text-slate-400" />

        <input
          type="date"
          value={startDate}
          min="2025-05-01"
          max="2025-08-31"
          onChange={(e) => setStartDate(e.target.value)}
          disabled={isActive}
          className="bg-slate-700 text-white text-sm rounded px-2 py-1 border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50 w-[130px]"
          title="Start date"
        />
        <span className="text-slate-500 text-xs">to</span>
        <input
          type="date"
          value={endDate}
          min={startDate}
          max="2025-08-31"
          onChange={(e) => setEndDate(e.target.value)}
          disabled={isActive}
          className="bg-slate-700 text-white text-sm rounded px-2 py-1 border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50 w-[130px]"
          title="End date"
        />
        {dayCount > 1 && !isActive && (
          <span className="text-xs text-slate-400">{dayCount}d</span>
        )}

        {/* Run */}
        {!isActive ? (
          <button
            onClick={handleRun}
            className="btn-primary flex items-center text-sm py-1.5 px-3"
            title={`Run simulation for ${dayCount} day(s)`}
          >
            <Play className="w-4 h-4 mr-1" />
            Run
          </button>
        ) : playbackFetching ? (
          <button className="btn-secondary flex items-center text-sm py-1.5 px-3 cursor-wait" disabled>
            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            Loading...
          </button>
        ) : null}

        {/* Pause / Resume */}
        {isActive && !playbackFetching && (
          <button
            onClick={handlePause}
            className="btn-secondary flex items-center text-sm py-1.5 px-3"
            title={playbackPaused ? 'Resume playback' : 'Pause playback'}
          >
            {playbackPaused ? (
              <>
                <Play className="w-4 h-4 mr-1" />
                Resume
              </>
            ) : (
              <>
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </>
            )}
          </button>
        )}

        {/* Stop */}
        {isActive && (
          <button
            onClick={handleStop}
            className="btn-danger flex items-center text-sm py-1.5 px-3"
            title="Stop simulation"
          >
            <Square className="w-4 h-4 mr-1" />
            Stop
          </button>
        )}

        {/* Skip */}
        {isActive && !playbackFetching && (
          <button
            onClick={handleSkip}
            className="btn-secondary flex items-center text-sm py-1.5 px-3"
            title="Advance one 15-min step"
          >
            <SkipForward className="w-4 h-4 mr-1" />
            Skip
          </button>
        )}

        {/* Speed selector */}
        {isActive && (
          <div className="flex items-center space-x-1 ml-1">
            <Gauge className="w-3.5 h-3.5 text-slate-400" />
            {SPEED_OPTIONS.map((opt) => (
              <button
                key={opt.ms}
                onClick={() => handleSpeedChange(opt.ms)}
                className={`px-1.5 py-0.5 text-[10px] rounded transition-colors ${
                  playbackSpeed === opt.ms
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}

        {error && (
          <span className="text-xs text-red-400 max-w-[200px] truncate" title={error}>
            {error}
          </span>
        )}
      </div>

      {/* Center: Live metrics during playback */}
      <div className="flex items-center space-x-5">
        {isActive && (
          <div className="flex items-center">
            <Activity className={`w-4 h-4 ${playbackPaused ? 'text-amber-400' : 'text-green-400 animate-pulse'}`} />
            <span className={`ml-1 text-xs ${playbackPaused ? 'text-amber-400' : 'text-green-400'}`}>
              {playbackFetching ? 'LOADING' : playbackPaused ? 'PAUSED' : 'RUNNING'}
            </span>
          </div>
        )}

        {isActive && playbackCurrentDate && (
          <div className="text-center">
            <p className="text-[10px] text-slate-500">Date</p>
            <p className="text-xs font-mono text-white">{playbackCurrentDate}</p>
          </div>
        )}

        {isActive && currentHourLabel && (
          <div className="text-center">
            <p className="text-[10px] text-slate-500">Time</p>
            <p className="text-xs font-mono text-white">{currentHourLabel}</p>
          </div>
        )}

        {isActive && (
          <div className="text-center">
            <p className="text-[10px] text-slate-500">Step</p>
            <p className="text-xs font-mono text-white">
              {playbackStepIndex}/96
              {playbackTotalDays > 1 && (
                <span className="text-slate-400 ml-1">(Day {playbackDayIndex + 1}/{playbackTotalDays})</span>
              )}
            </p>
          </div>
        )}

        {/* Show live power metrics during playback */}
        {liveMetrics && (
          <>
            <div className="text-center">
              <p className="text-[10px] text-slate-500">Load</p>
              <p className="text-xs font-semibold text-white">
                {(liveMetrics.total_power_kw / 1000).toFixed(2)} MW
              </p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-slate-500">Solar</p>
              <p className={`text-xs font-semibold ${liveMetrics.total_solar_kw > 0 ? 'text-yellow-400' : 'text-slate-500'}`}>
                {(liveMetrics.total_solar_kw / 1000).toFixed(2)} MW
              </p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-slate-500">Losses</p>
              <p className="text-xs font-semibold text-amber-400">
                {liveMetrics.total_losses_kw.toFixed(1)} kW
              </p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-slate-500">Violations</p>
              <p className={`text-xs font-semibold ${liveMetrics.voltage_violations > 0 ? 'text-red-400' : 'text-green-400'}`}>
                {liveMetrics.voltage_violations}
              </p>
            </div>
          </>
        )}

        {/* Show static grid stats when NOT playing back */}
        {!isActive && gridState && (
          <>
            <div className="text-center">
              <p className="text-xs text-slate-400">Load</p>
              <p className="text-sm font-semibold text-white">
                {gridState.summary.total_load_kw.toFixed(1)} kW
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-400">Generation</p>
              <p className="text-sm font-semibold text-green-400">
                {gridState.summary.total_generation_kw.toFixed(1)} kW
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-400">Losses</p>
              <p className="text-sm font-semibold text-amber-400">
                {gridState.summary.total_losses_kw.toFixed(2)} kW
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-slate-400">Violations</p>
              <p className={`text-sm font-semibold ${
                gridState.summary.num_voltage_violations > 0 ? 'text-red-400' : 'text-green-400'
              }`}>
                {gridState.summary.num_voltage_violations}
              </p>
            </div>
          </>
        )}
      </div>

      {/* Right: Convergence badge */}
      <div className="flex items-center space-x-4">
        {liveMetrics && (
          <div className={`px-3 py-1 rounded-full text-xs ${
            liveMetrics.converged ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
          }`}>
            {liveMetrics.converged ? 'Converged' : 'Not Converged'}
          </div>
        )}
        {!isActive && gridState?.converged !== undefined && (
          <div className={`px-3 py-1 rounded-full text-sm ${
            gridState.converged ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
          }`}>
            {gridState.converged ? 'Converged' : 'Not Converged'}
          </div>
        )}
      </div>
    </header>
  );
}
