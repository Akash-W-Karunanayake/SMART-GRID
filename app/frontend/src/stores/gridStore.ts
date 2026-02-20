/**
 * Zustand store for grid state management
 */

import { create } from 'zustand';
import type { GridState, SimulationStatus, SimulationHistoryItem, Topology } from '../types';

/** Per-step pipeline data used by Dashboard charts */
export interface PipelineStep {
  hour: number;
  total_power_kw: number;
  total_losses_kw: number;
  min_voltage_pu: number;
  max_voltage_pu: number;
  voltage_violations: number;
  converged: boolean;
  total_solar_kw: number;
  total_wind_kw: number;
  total_thermal_kw: number;
  total_generation_kw: number;
  /** Per-bus voltage (pu) keyed by bus name — populated from pipeline */
  bus_voltages?: Record<string, number>;
  /** Per-feeder net power (kW) — positive = import, negative = export/reverse flow */
  power_F06_kw?: number;
  power_F07_kw?: number;
  power_F08_kw?: number;
  power_F09_kw?: number;
  power_F10_kw?: number;
  power_F11_kw?: number;
  power_F12_kw?: number;
}

/** Live metrics displayed during step-by-step playback */
export interface LiveMetrics {
  hour: number;
  total_power_kw: number;
  total_losses_kw: number;
  min_voltage_pu: number;
  max_voltage_pu: number;
  voltage_violations: number;
  converged: boolean;
  total_solar_kw: number;
  total_wind_kw: number;
  total_thermal_kw: number;
  total_generation_kw: number;
  /** Per-bus voltage (pu) keyed by bus name — enables topology node coloring per step */
  bus_voltages?: Record<string, number>;
  /** Per-feeder net power (kW) */
  power_F06_kw?: number;
  power_F07_kw?: number;
  power_F08_kw?: number;
  power_F09_kw?: number;
  power_F10_kw?: number;
  power_F11_kw?: number;
  power_F12_kw?: number;
}

interface GridStore {
  // State
  gridState: GridState | null;
  simulationStatus: SimulationStatus | null;
  history: SimulationHistoryItem[];
  topology: Topology | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  /** Pipeline 96-step data from the most recent single-day simulation */
  pipelineSteps: PipelineStep[];
  /** Date of the last pipeline simulation */
  lastSimDate: string | null;

  // ─── Playback state ───────────────────────────────────
  /** Whether playback is actively running (timer is ticking) */
  playbackPlaying: boolean;
  /** Whether playback is paused (timer stopped, can resume) */
  playbackPaused: boolean;
  /** Index of current day being played (0-based, within the date range) */
  playbackDayIndex: number;
  /** Total number of days in this simulation run */
  playbackTotalDays: number;
  /** Index of current step within the current day (0-95) */
  playbackStepIndex: number;
  /** Current date being simulated */
  playbackCurrentDate: string | null;
  /** The accumulated pipeline steps shown so far for the current day */
  playbackVisibleSteps: PipelineStep[];
  /** Delay between steps in milliseconds */
  playbackSpeed: number;
  /** Whether the backend is currently being fetched (loading a day) */
  playbackFetching: boolean;
  /** Live step metrics updated each tick during playback */
  liveMetrics: LiveMetrics | null;

  // Actions
  setGridState: (state: GridState) => void;
  setSimulationStatus: (status: SimulationStatus) => void;
  addHistoryItem: (item: SimulationHistoryItem) => void;
  setHistory: (history: SimulationHistoryItem[]) => void;
  setTopology: (topology: Topology) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setPipelineSteps: (steps: PipelineStep[], date: string) => void;

  // Playback actions
  startPlayback: (totalDays: number, firstDate: string) => void;
  setPlaybackDay: (dayIndex: number, date: string) => void;
  advancePlaybackStep: (step: PipelineStep) => void;
  pausePlayback: () => void;
  resumePlayback: () => void;
  stopPlayback: () => void;
  setPlaybackSpeed: (ms: number) => void;
  setPlaybackFetching: (fetching: boolean) => void;
  finishPlayback: () => void;

  clearState: () => void;
}

const initialPlaybackState = {
  playbackPlaying: false,
  playbackPaused: false,
  playbackDayIndex: 0,
  playbackTotalDays: 0,
  playbackStepIndex: 0,
  playbackCurrentDate: null as string | null,
  playbackVisibleSteps: [] as PipelineStep[],
  playbackSpeed: 400,
  playbackFetching: false,
  liveMetrics: null as LiveMetrics | null,
};

export const useGridStore = create<GridStore>((set) => ({
  // Initial state
  gridState: null,
  simulationStatus: null,
  history: [],
  topology: null,
  isConnected: false,
  isLoading: false,
  error: null,
  pipelineSteps: [],
  lastSimDate: null,
  ...initialPlaybackState,

  // Actions
  setGridState: (state) => set({ gridState: state }),

  setSimulationStatus: (status) => set({ simulationStatus: status }),

  addHistoryItem: (item) =>
    set((state) => ({
      history: [...state.history.slice(-999), item],
    })),

  setHistory: (history) => set({ history }),

  setTopology: (topology) => set({ topology }),

  setConnected: (connected) => set({ isConnected: connected }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  setPipelineSteps: (steps, date) => set({ pipelineSteps: steps, lastSimDate: date }),

  // ─── Playback actions ───────────────────────────────────

  startPlayback: (totalDays, firstDate) =>
    set({
      playbackPlaying: true,
      playbackPaused: false,
      playbackDayIndex: 0,
      playbackTotalDays: totalDays,
      playbackStepIndex: 0,
      playbackCurrentDate: firstDate,
      playbackVisibleSteps: [],
      playbackFetching: false,
      liveMetrics: null,
      error: null,
    }),

  setPlaybackDay: (dayIndex, date) =>
    set({
      playbackDayIndex: dayIndex,
      playbackCurrentDate: date,
      playbackStepIndex: 0,
      playbackVisibleSteps: [],
      liveMetrics: null,
    }),

  advancePlaybackStep: (step) =>
    set((state) => ({
      playbackStepIndex: state.playbackStepIndex + 1,
      playbackVisibleSteps: [...state.playbackVisibleSteps, step],
      liveMetrics: {
        hour: step.hour,
        total_power_kw: step.total_power_kw,
        total_losses_kw: step.total_losses_kw,
        min_voltage_pu: step.min_voltage_pu,
        max_voltage_pu: step.max_voltage_pu,
        voltage_violations: step.voltage_violations,
        converged: step.converged,
        total_solar_kw: step.total_solar_kw,
        total_wind_kw: step.total_wind_kw,
        total_thermal_kw: step.total_thermal_kw,
        total_generation_kw: step.total_generation_kw,
        bus_voltages: step.bus_voltages,
        power_F06_kw: step.power_F06_kw,
        power_F07_kw: step.power_F07_kw,
        power_F08_kw: step.power_F08_kw,
        power_F09_kw: step.power_F09_kw,
        power_F10_kw: step.power_F10_kw,
        power_F11_kw: step.power_F11_kw,
        power_F12_kw: step.power_F12_kw,
      },
    })),

  pausePlayback: () => set({ playbackPaused: true }),

  resumePlayback: () => set({ playbackPaused: false }),

  stopPlayback: () => set({ ...initialPlaybackState }),

  setPlaybackSpeed: (ms) => set({ playbackSpeed: ms }),

  setPlaybackFetching: (fetching) => set({ playbackFetching: fetching }),

  finishPlayback: () =>
    set({
      playbackPlaying: false,
      playbackPaused: false,
      playbackFetching: false,
    }),

  clearState: () =>
    set({
      gridState: null,
      simulationStatus: null,
      history: [],
      pipelineSteps: [],
      lastSimDate: null,
      error: null,
      ...initialPlaybackState,
    }),
}));

export default useGridStore;
