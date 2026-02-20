/**
 * Simulation playback hook.
 *
 * Fetches all 96 steps for each day in a date range via `simulateSingleDay`,
 * then plays them back one step at a time on a timer so the Dashboard
 * updates in real-time (configurable speed, default ~400 ms per step).
 *
 * Supports: run (single or range), pause, resume, stop, skip (next step).
 */

import { useRef, useCallback, useEffect } from 'react';
import { useGridStore, type PipelineStep } from '../stores/gridStore';
import api from '../services/api';
import type { GridState } from '../types';

/** Generate array of ISO date strings between start and end (inclusive). */
function dateRange(start: string, end: string): string[] {
  const dates: string[] = [];
  const cur = new Date(start + 'T00:00:00');
  const last = new Date(end + 'T00:00:00');
  while (cur <= last) {
    // Use local date parts to avoid UTC timezone shift (e.g. IST UTC+5:30)
    const yyyy = cur.getFullYear();
    const mm = String(cur.getMonth() + 1).padStart(2, '0');
    const dd = String(cur.getDate()).padStart(2, '0');
    dates.push(`${yyyy}-${mm}-${dd}`);
    cur.setDate(cur.getDate() + 1);
  }
  return dates;
}

/** Shape a raw backend step object into our PipelineStep interface. */
function toStep(s: any): PipelineStep {
  return {
    hour: s.hour ?? 0,
    total_power_kw: s.total_power_kw ?? 0,
    total_losses_kw: s.total_losses_kw ?? 0,
    min_voltage_pu: s.min_voltage_pu ?? 0,
    max_voltage_pu: s.max_voltage_pu ?? 0,
    voltage_violations: s.voltage_violations ?? 0,
    converged: s.converged ?? false,
    total_solar_kw: s.total_solar_kw ?? 0,
    total_wind_kw: s.total_wind_kw ?? 0,
    total_thermal_kw: s.total_thermal_kw ?? 0,
    total_generation_kw: s.total_generation_kw ?? 0,
    bus_voltages: s.bus_voltages,
    power_F06_kw: s.power_F06_kw,
    power_F07_kw: s.power_F07_kw,
    power_F08_kw: s.power_F08_kw,
    power_F09_kw: s.power_F09_kw,
    power_F10_kw: s.power_F10_kw,
    power_F11_kw: s.power_F11_kw,
    power_F12_kw: s.power_F12_kw,
  };
}

export function useSimulationPlayback() {
  // Refs so the timer callback always reads fresh state without re-creating.
  const stepsRef = useRef<PipelineStep[]>([]);
  const indexRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const datesRef = useRef<string[]>([]);
  const dayIdxRef = useRef(0);
  const cancelledRef = useRef(false);

  // ─── cleanup on unmount ───────────────────────────────
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // ─── tick: play one step ──────────────────────────────
  const tick = useCallback(() => {
    if (cancelledRef.current) return;
    const state = useGridStore.getState();
    if (!state.playbackPlaying || state.playbackPaused) return;

    const steps = stepsRef.current;
    const idx = indexRef.current;

    if (idx >= steps.length) {
      // Current day finished – move to next day or finish
      dayIdxRef.current += 1;
      if (dayIdxRef.current < datesRef.current.length) {
        fetchAndPlayDay(dayIdxRef.current);
      } else {
        // All days done
        state.finishPlayback();
      }
      return;
    }

    // Advance one step
    const step = steps[idx];
    state.advancePlaybackStep(step);
    indexRef.current = idx + 1;

    // Schedule next tick
    timerRef.current = setTimeout(tick, state.playbackSpeed);
  }, []);

  // ─── fetchAndPlayDay: get data for one day, start ticking ─
  const fetchAndPlayDay = useCallback(
    async (dayIndex: number) => {
      if (cancelledRef.current) return;
      const state = useGridStore.getState();
      const date = datesRef.current[dayIndex];
      if (!date) {
        state.finishPlayback();
        return;
      }

      state.setPlaybackFetching(true);
      state.setPlaybackDay(dayIndex, date);

      try {
        const result = await api.simulateSingleDay(date);

        if (cancelledRef.current) return;

        // Update grid state + topology once per day
        if (result.grid_state) {
          state.setGridState(result.grid_state as unknown as GridState);
        }

        // Store the full 96 steps for the playback timer
        const allSteps = (result.steps as any[]).map(toStep);
        stepsRef.current = allSteps;
        indexRef.current = 0;

        // Also push them into store so they're available after playback ends
        state.setPipelineSteps(allSteps, date);

        state.setPlaybackFetching(false);

        // Start ticking
        if (!cancelledRef.current && useGridStore.getState().playbackPlaying) {
          timerRef.current = setTimeout(tick, useGridStore.getState().playbackSpeed);
        }
      } catch (e: unknown) {
        if (!cancelledRef.current) {
          const msg = e instanceof Error ? e.message : 'Simulation failed';
          useGridStore.getState().setError(msg);
          useGridStore.getState().stopPlayback();
        }
      }
    },
    [tick],
  );

  // ─── public: run ──────────────────────────────────────
  const run = useCallback(
    (startDate: string, endDate: string) => {
      // Stop any existing playback
      if (timerRef.current) clearTimeout(timerRef.current);
      cancelledRef.current = false;

      const dates = dateRange(startDate, endDate);
      datesRef.current = dates;
      dayIdxRef.current = 0;

      const state = useGridStore.getState();
      state.startPlayback(dates.length, dates[0]);

      fetchAndPlayDay(0);
    },
    [fetchAndPlayDay],
  );

  // ─── public: pause ────────────────────────────────────
  const pause = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    useGridStore.getState().pausePlayback();
  }, []);

  // ─── public: resume ───────────────────────────────────
  const resume = useCallback(() => {
    useGridStore.getState().resumePlayback();
    timerRef.current = setTimeout(tick, useGridStore.getState().playbackSpeed);
  }, [tick]);

  // ─── public: stop ─────────────────────────────────────
  const stop = useCallback(() => {
    cancelledRef.current = true;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    useGridStore.getState().stopPlayback();
  }, []);

  // ─── public: skip (advance one step manually) ─────────
  const skip = useCallback(() => {
    const state = useGridStore.getState();
    if (!state.playbackPlaying) return;

    // If timer is running, clear it first so we don't double-advance
    if (timerRef.current) clearTimeout(timerRef.current);

    const steps = stepsRef.current;
    const idx = indexRef.current;

    if (idx >= steps.length) {
      // Move to next day
      dayIdxRef.current += 1;
      if (dayIdxRef.current < datesRef.current.length) {
        fetchAndPlayDay(dayIdxRef.current);
      } else {
        state.finishPlayback();
      }
      return;
    }

    const step = steps[idx];
    state.advancePlaybackStep(step);
    indexRef.current = idx + 1;

    // If not paused, restart the timer
    if (!state.playbackPaused) {
      timerRef.current = setTimeout(tick, state.playbackSpeed);
    }
  }, [tick, fetchAndPlayDay]);

  // ─── public: setSpeed ─────────────────────────────────
  const setSpeed = useCallback((ms: number) => {
    useGridStore.getState().setPlaybackSpeed(ms);
  }, []);

  return { run, pause, resume, stop, skip, setSpeed };
}

export default useSimulationPlayback;
