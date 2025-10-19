/**
 * TypeScript types for GenML Pipeline data structures.
 * These match the JSON structures produced by the Python backend.
 */

export type PipelineStatus = 'idle' | 'running' | 'completed' | 'failed';
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed';
export type ModelStatus = 'pending' | 'training' | 'completed' | 'failed';

export interface StageInfo {
  name: string;
  description?: string;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  summary: Record<string, any>;
}

export interface TrialHistory {
  trial: number;
  score: number;
  timestamp: string;
}

export interface ModelResult {
  name: string;
  status: ModelStatus;
  started_at?: string;
  completed_at?: string;
  mean_score: number | null;
  std_score: number | null;
  best_params: Record<string, any>;
  tuned: boolean;
  current_trial?: number;
  total_trials?: number | null;
  best_score?: number | null;
  trial_history?: TrialHistory[];
}

export interface ResourceMetrics {
  gpu_memory_mb: number;
  gpu_memory_total_mb: number;
  cpu_percent: number;
  ram_mb: number;
  elapsed_seconds: number;
}

export interface AIInsights {
  model_selection: Record<string, any>;
  feature_suggestions: any[];
  error_patterns: any[];
}

export interface RunProgress {
  run_id: string | null;
  dataset_name?: string;
  status: PipelineStatus;
  started_at?: string;
  completed_at?: string | null;
  current_stage?: number;
  current_stage_name?: string;
  stage_progress_pct?: number;
  current_task?: string;
  eta_seconds?: number | null;
  stages?: Record<string, StageInfo>;
  models?: ModelResult[];
  resources?: ResourceMetrics;
  ai_insights?: AIInsights;
  message?: string;  // For idle state message
}

export interface RunSummary {
  run_id: string;
  dataset_name: string;
  status: PipelineStatus;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
}

export interface Report {
  name: string;
  filename: string;
  size_bytes: number;
}

export interface ModelComparison {
  problem_type: string;
  best_model: string;
  best_score: number;
  scoring_metric: string;
  all_results: Record<string, any>;
  gpu_used: boolean;
}

export interface LogData {
  run_id: string;
  filename: string;
  content: string;
  lines: number;
}
