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

  async forecastSolar(horizonHours: number = 24, includeUncertainty: boolean = true, targetDate?: string) {
    const params = new URLSearchParams();
    if (targetDate) params.set('target_date', targetDate);
    const qs = params.toString();
    return this.request(`/forecasting/solar${qs ? '?' + qs : ''}`, {
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

  async getSolarDayData(date: string) {
    return this.request(`/forecasting/solar-day-data?date=${date}`);
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

  async getModelStatus() {
    return this.request<{ loaded: boolean; dss_model_loaded: boolean; can_inject: boolean; n_buses: number | null; n_features_cnn: number | null }>(
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

  // ============== Self-Healing API ==============

  async runFLISR(
    faultBus: string,
    faultType: string = '3phase',
    faultResistance: number = 0.01,
    restorationStrategy: string = 'auto'
  ) {
    return this.request<import('../types').FLISRResponse>('/self-healing/flisr', {
      method: 'POST',
      body: JSON.stringify({
        fault_bus: faultBus,
        fault_type: faultType,
        fault_resistance: faultResistance,
        restoration_strategy: restorationStrategy,
      }),
    });
  }

  async isolateFault(faultBus: string, faultType: string = '3phase', faultResistance: number = 0.01) {
    return this.request<import('../types').IsolationResult>('/self-healing/isolate', {
      method: 'POST',
      body: JSON.stringify({
        fault_type: faultType,
        fault_location: faultBus,
        fault_resistance_ohms: faultResistance,
      }),
    });
  }

  async restoreService(strategy: string = 'auto') {
    return this.request<import('../types').RestorationResult>('/self-healing/restore', {
      method: 'POST',
      body: JSON.stringify({ strategy }),
    });
  }

  async getSwitchStates() {
    return this.request<{ switches: Record<string, boolean> }>('/self-healing/switches');
  }

  async controlSwitch(name: string, closed: boolean) {
    return this.request<{ switch: string; closed: boolean; converged: boolean; message: string }>(
      `/self-healing/switch/${name}`,
      {
        method: 'PUT',
        body: JSON.stringify({ closed }),
      }
    );
  }

  async getSelfHealingGridState() {
    return this.request<import('../types').SelfHealingGridState>('/self-healing/grid-state');
  }

  async resetGrid() {
    return this.request<{ success: boolean; message: string }>('/self-healing/reset', {
      method: 'POST',
    });
  }

  // ============== Health Check ==============

  async healthCheck() {
    return this.request('/health');
  }
}

export const api = new ApiService();
export default api;
