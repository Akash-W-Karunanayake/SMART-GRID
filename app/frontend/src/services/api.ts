/**
 * API Service for Smart Grid AI Framework
 * Handles all HTTP requests to the backend
 */

const API_BASE_URL = '/api/v1';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // ============== Grid API ==============

  async loadModel() {
    return this.request('/grid/load', { method: 'POST' });
  }

  async getGridState() {
    return this.request('/grid/state');
  }

  async getTopology() {
    return this.request('/grid/topology');
  }

  async getVoltageProfile() {
    return this.request('/grid/voltage-profile');
  }

  async setLoadMultiplier(multiplier: number) {
    return this.request('/grid/load-multiplier', {
      method: 'POST',
      body: JSON.stringify({ multiplier }),
    });
  }

  async setGenerationMultiplier(multiplier: number) {
    return this.request('/grid/generation-multiplier', {
      method: 'POST',
      body: JSON.stringify({ multiplier }),
    });
  }

  async getCircuitInfo() {
    return this.request('/grid/info');
  }

  async getAllBuses() {
    return this.request<{ buses: string[] }>('/grid/buses');
  }

  async getAllElements() {
    return this.request<{ elements: string[] }>('/grid/elements');
  }

  // ============== Simulation API ==============

  async startSimulation(hours: number = 24, speed: number = 1.0) {
    return this.request('/simulation/start', {
      method: 'POST',
      body: JSON.stringify({ hours, speed }),
    });
  }

  async stopSimulation() {
    return this.request('/simulation/stop', { method: 'POST' });
  }

  async pauseSimulation() {
    return this.request('/simulation/pause', { method: 'POST' });
  }

  async resumeSimulation() {
    return this.request('/simulation/resume', { method: 'POST' });
  }

  async stepSimulation() {
    return this.request('/simulation/step', { method: 'POST' });
  }

  async setSimulationSpeed(speed: number) {
    return this.request(`/simulation/speed?speed=${speed}`, { method: 'POST' });
  }

  async getSimulationStatus() {
    return this.request('/simulation/status');
  }

  async getSimulationHistory(limit: number = 100) {
    return this.request(`/simulation/history?limit=${limit}`);
  }

  async getCurrentState() {
    return this.request('/simulation/current-state');
  }

  // ============== Forecasting API ==============

  async forecastLoad(horizonHours: number = 24, includeUncertainty: boolean = true) {
    return this.request('/forecasting/load', {
      method: 'POST',
      body: JSON.stringify({ horizon_hours: horizonHours, include_uncertainty: includeUncertainty }),
    });
  }

  async forecastSolar(horizonHours: number = 24, includeUncertainty: boolean = true) {
    return this.request('/forecasting/solar', {
      method: 'POST',
      body: JSON.stringify({ horizon_hours: horizonHours, include_uncertainty: includeUncertainty }),
    });
  }

  async forecastNetLoad(horizonHours: number = 24, includeUncertainty: boolean = true) {
    return this.request('/forecasting/net-load', {
      method: 'POST',
      body: JSON.stringify({ horizon_hours: horizonHours, include_uncertainty: includeUncertainty }),
    });
  }

  async detectImbalance() {
    return this.request('/forecasting/imbalance-detection');
  }

  async getHouseholdAlerts() {
    return this.request('/forecasting/household-alerts');
  }

  async getGridOperatorData() {
    return this.request('/forecasting/grid-operator-dashboard');
  }

  // ============== Diagnostics API ==============

  async injectFault(bus: string, faultType: string, phase: string, resistance: number = 1.0) {
    return this.request<{ success: boolean; message: string }>('/diagnostics/inject-fault', {
      method: 'POST',
      body: JSON.stringify({ bus, fault_type: faultType, phase, resistance }),
    });
  }

  async clearFault() {
    return this.request<{ success: boolean; message: string }>('/diagnostics/clear-fault', {
      method: 'POST',
    });
  }

  async getFaultStatus() {
    return this.request<{
      has_active_fault: boolean;
      active_fault?: { bus: string; fault_type: string; phase: string; resistance: number; step_injected: number };
      latest_prediction?: Record<string, unknown>;
      detection_latency_steps?: number;
    }>('/diagnostics/fault-status');
  }

  async getFaultHistory() {
    return this.request<Array<Record<string, unknown>>>('/diagnostics/fault-history');
  }

  async getFaultWaveform() {
    return this.request<{
      voltage_seq: number[][][];
      current_seq: number[][][];
      bus_names: string[];
      branch_names: string[];
      total_cycles: number;
    }>('/diagnostics/fault-waveform');
  }

  async getModelStatus() {
    return this.request<{ loaded: boolean; n_buses: number | null; n_features_cnn: number | null }>(
      '/diagnostics/model-status'
    );
  }

  async loadFaultModel() {
    return this.request<{ success: boolean; message: string }>('/diagnostics/load-model', {
      method: 'POST',
    });
  }

  // ============== Pipeline Simulation API ==============

  async startPipelineSimulation(startDate: string, endDate?: string) {
    return this.request<{ task_id: string; mode: string; total_days: number; message: string }>(
      '/pipeline/simulate',
      {
        method: 'POST',
        body: JSON.stringify({ start_date: startDate, end_date: endDate }),
      }
    );
  }

  async getPipelineStatus(taskId: string) {
    return this.request<{
      task_id: string; status: string; mode: string;
      total_days: number; current_day: number; current_date: string;
      completed_count: number; error?: string;
    }>(`/pipeline/status/${taskId}`);
  }

  async getPipelineResults(taskId: string) {
    return this.request<{
      task_id: string; status: string; mode: string;
      start_date: string; end_date: string; total_days: number;
      completed_days: Record<string, unknown>[];
    }>(`/pipeline/results/${taskId}`);
  }

  async simulateSingleDay(date: string) {
    return this.request<{
      summary: Record<string, unknown>;
      steps: Record<string, unknown>[];
      grid_state: Record<string, unknown>;
    }>(
      '/pipeline/simulate-day',
      {
        method: 'POST',
        body: JSON.stringify({ date }),
      }
    );
  }

  async getGridStateNoSolve() {
    return this.request('/grid/current-state');
  }

  async cancelPipelineTask(taskId: string) {
    return this.request(`/pipeline/cancel/${taskId}`, { method: 'POST' });
  }

  // ============== Health Check ==============

  async healthCheck() {
    return this.request('/health');
  }
}

export const api = new ApiService();
export default api;
