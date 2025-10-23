/**
 * API Client for GenML Pipeline Backend.
 * Handles all HTTP requests to the FastAPI server.
 */

import axios from 'axios';
import type {
  RunProgress,
  RunSummary,
  Report,
  ModelComparison,
  LogData,
} from '../types/pipeline';

// Base URL for API (proxied through Vite in development)
const API_BASE_URL = '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  /**
   * Health check
   */
  health: async (): Promise<{ status: string; service: string }> => {
    const { data } = await apiClient.get('/health');
    return data;
  },

  /**
   * Get current pipeline status
   */
  getStatus: async (): Promise<RunProgress> => {
    const { data } = await apiClient.get('/status');
    return data;
  },

  /**
   * Get list of all runs
   */
  getRuns: async (): Promise<RunSummary[]> => {
    const { data } = await apiClient.get('/runs');
    return data;
  },

  /**
   * Get specific run details
   */
  getRun: async (runId: string): Promise<RunProgress> => {
    const { data } = await apiClient.get(`/runs/${runId}`);
    return data;
  },

  /**
   * List available reports
   */
  listReports: async (): Promise<Report[]> => {
    const { data } = await apiClient.get('/reports');
    return data;
  },

  /**
   * Get specific report
   */
  getReport: async (reportName: string): Promise<any> => {
    const { data } = await apiClient.get(`/reports/${reportName}`);
    return data;
  },

  /**
   * Get log file for a run
   */
  getLog: async (runId: string): Promise<LogData> => {
    const { data } = await apiClient.get(`/logs/${runId}`);
    return data;
  },

  /**
   * Get model comparison data
   */
  getModelsComparison: async (): Promise<ModelComparison> => {
    const { data } = await apiClient.get('/models');
    return data;
  },
};

export default api;
