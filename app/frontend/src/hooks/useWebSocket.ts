/**
 * Custom hook for WebSocket connection management
 */

import { useEffect, useCallback } from 'react';
import { wsService } from '../services/websocket';
import { useGridStore } from '../stores/gridStore';
import type { WSMessage, GridState } from '../types';

export function useWebSocket() {
  const {
    setGridState,
    setSimulationStatus,
    addHistoryItem,
    setConnected,
    setError,
  } = useGridStore();

  // Handle incoming messages
  const handleMessage = useCallback((message: WSMessage) => {
    switch (message.type) {
      case 'state_update':
        if (message.data) {
          const state = message.data as GridState;
          setGridState(state);
          // Add to history for Power Flow Over Time chart
          addHistoryItem({
            timestamp: state.timestamp,
            total_power_kw: state.summary.total_power_kw,
            total_load_kw: state.summary.total_load_kw,
            total_generation_kw: state.summary.total_generation_kw,
            total_losses_kw: state.summary.total_losses_kw,
            converged: state.converged,
            num_violations: state.summary.num_voltage_violations,
          });
        }
        break;
      case 'status':
        if (message.data) {
          setSimulationStatus(message.data as any);
        }
        break;
      case 'response':
        // Handle control action responses (start/stop/pause/resume/step)
        // Request fresh status to ensure UI is in sync
        wsService.getStatus();
        break;
      case 'error':
        setError(message.message || 'Unknown error');
        break;
      case 'info':
        console.log('WebSocket info:', message.message);
        break;
    }
  }, [setGridState, setSimulationStatus, setError, addHistoryItem]);

  // Connect on mount
  useEffect(() => {
    const connect = async () => {
      try {
        await wsService.connect();
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setError('Failed to connect to server');
      }
    };

    connect();

    // Subscribe to events
    const unsubMessage = wsService.onMessage(handleMessage);
    const unsubConnect = wsService.onConnect(() => setConnected(true));
    const unsubDisconnect = wsService.onDisconnect(() => setConnected(false));

    // Cleanup on unmount
    return () => {
      unsubMessage();
      unsubConnect();
      unsubDisconnect();
    };
  }, [handleMessage, setConnected, setError]);

  // Control functions
  const startSimulation = useCallback((hours: number = 24, speed: number = 1.0) => {
    wsService.startSimulation(hours, speed);
  }, []);

  const stopSimulation = useCallback(() => {
    wsService.stopSimulation();
  }, []);

  const pauseSimulation = useCallback(() => {
    wsService.pauseSimulation();
  }, []);

  const resumeSimulation = useCallback(() => {
    wsService.resumeSimulation();
  }, []);

  const stepSimulation = useCallback(() => {
    wsService.stepSimulation();
  }, []);

  const setSpeed = useCallback((speed: number) => {
    wsService.setSpeed(speed);
  }, []);

  const requestState = useCallback(() => {
    wsService.getState();
  }, []);

  const requestStatus = useCallback(() => {
    wsService.getStatus();
  }, []);

  return {
    isConnected: wsService.isConnected,
    startSimulation,
    stopSimulation,
    pauseSimulation,
    resumeSimulation,
    stepSimulation,
    setSpeed,
    requestState,
    requestStatus,
  };
}

export default useWebSocket;
