// Grid State Types
export interface BusData {
  name: string;
  base_kv: number;
  voltage_pu: number[];
  voltage_angle: number[];
}

export interface LineData {
  name: string;
  bus1: string;
  bus2: string;
  power_kw: number;
  current_amps: number[];
  current_angle: number[];
  enabled: boolean;
}

export interface TransformerData {
  name: string;
  kva: number;
  loading_percent: number;
  power_kw: number;
}

export interface LoadData {
  name: string;
  bus: string;
  kw: number;
  kvar: number;
  voltage_pu: number;
}

export interface GeneratorData {
  name: string;
  bus: string;
  kw: number;
  kvar: number;
  type: string;
}

export interface GridSummary {
  total_power_kw: number;
  total_power_kvar: number;
  total_losses_kw: number;
  total_generation_kw: number;
  total_solar_kw: number;
  total_load_kw: number;
  num_voltage_violations: number;
  num_overloaded_elements: number;
}

export interface Violations {
  voltage: string[];
  overloads: string[];
}

export interface FaultPrediction {
  is_fault: boolean;
  detection_confidence: number;
  fault_type: string;
  type_probabilities: Record<string, number>;
  fault_phase: string;
  phase_probabilities: Record<string, number>;
  fault_location_bus: string;
  location_probabilities: Record<string, number>;
  step_injected?: number;
  step_detected?: number;
}

export interface ActiveFault {
  bus: string;
  fault_type: string;
  phase: string;
  resistance: number;
  step_injected: number;
}

export interface FaultPayload {
  has_active_fault: boolean;
  active_fault?: ActiveFault;
  prediction?: FaultPrediction;
  detection_latency_steps?: number;
}

export interface SubCycleWaveform {
  voltage_seq: number[][][];  // [20][N_buses][6]
  current_seq: number[][][];  // [20][N_branches][6]
  bus_names: string[];
  branch_names: string[];
  total_cycles: number;
}

export interface GridState {
  timestamp: number;
  simulation_time: string;
  converged: boolean;
  summary: GridSummary;
  buses: Record<string, BusData>;
  lines: Record<string, LineData>;
  transformers: Record<string, TransformerData>;
  loads: Record<string, LoadData>;
  generators: Record<string, GeneratorData>;
  violations: Violations;
  fault?: FaultPayload;
}

// Topology Types
export interface TopologyNode {
  id: string;
  label: string;
  type: string;
  kv?: number;
  x?: number;
  y?: number;
}

export interface TopologyEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  label: string;
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

// Simulation Types
export interface SimulationStatus {
  running: boolean;
  paused: boolean;
  current_hour: number;
  speed: number;
  subscribers: number;
  history_length: number;
  model_loaded: boolean;
}

export interface SimulationHistoryItem {
  timestamp: number;
  total_power_kw: number;
  total_load_kw: number;
  total_generation_kw: number;
  total_losses_kw: number;
  converged: boolean;
  num_violations: number;
}

// Forecasting Types
export interface ForecastPoint {
  timestamp: number;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ForecastResponse {
  forecast_type: string;
  horizon_hours: number;
  points: ForecastPoint[];
  model_info?: Record<string, unknown>;
}

// WebSocket Message Types
export interface WSMessage {
  type: 'state_update' | 'status' | 'error' | 'info' | 'pong' | 'response' | 'history';
  data?: unknown;
  message?: string;
  timestamp?: string;
  action?: string;
}

export interface WSControlMessage {
  action: 'start' | 'stop' | 'pause' | 'resume' | 'step' | 'set_speed' | 'get_state' | 'get_status' | 'ping';
  params?: Record<string, unknown>;
}

// Pipeline Simulation Types
export interface PipelineTaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'cancelled';
  mode: 'single' | 'range';
  total_days: number;
  current_day: number;
  current_date: string;
  completed_count: number;
  error?: string;
}

export interface DayResult {
  date: string;
  status: string;
  converged_steps?: number;
  total_steps?: number;
  min_voltage_pu?: number;
  max_voltage_pu?: number;
  total_violations?: number;
  avg_power_kw?: number;
  peak_power_kw?: number;
  min_power_kw?: number;
  error?: string;
  [key: string]: unknown;
}

export interface SimulationStep {
  step: number;
  hour: number;
  converged: boolean;
  total_power_kw?: number;
  total_losses_kw?: number;
  min_voltage_pu?: number;
  max_voltage_pu?: number;
  voltage_violations?: number;
  [key: string]: unknown;
}

export interface SingleDayResult {
  summary: DayResult;
  steps: SimulationStep[];
}

export interface PipelineResults {
  task_id: string;
  status: string;
  mode: string;
  start_date: string;
  end_date: string;
  total_days: number;
  completed_days: DayResult[];
}

// API Response Types
export interface ApiResponse<T = unknown> {
  success: boolean;
  message?: string;
  error?: string;
  data?: T;
}
