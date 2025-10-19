/**
 * Hook for fetching and managing progress data.
 * Can fetch from API or use WebSocket for real-time updates.
 */

import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import type { RunProgress, RunSummary } from '../types/pipeline';

/**
 * Fetch current pipeline status
 */
export const useCurrentStatus = (enabled = true) => {
  return useQuery({
    queryKey: ['status'],
    queryFn: () => api.getStatus(),
    refetchInterval: enabled ? 2000 : false, // Poll every 2 seconds if enabled
    enabled,
  });
};

/**
 * Fetch list of all runs
 */
export const useRuns = () => {
  return useQuery<RunSummary[]>({
    queryKey: ['runs'],
    queryFn: () => api.getRuns(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });
};

/**
 * Fetch specific run details
 */
export const useRun = (runId: string | undefined) => {
  return useQuery<RunProgress>({
    queryKey: ['run', runId],
    queryFn: () => api.getRun(runId!),
    enabled: !!runId,
  });
};

/**
 * Fetch model comparison data
 */
export const useModelsComparison = () => {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => api.getModelsComparison(),
  });
};

/**
 * Fetch reports list
 */
export const useReports = () => {
  return useQuery({
    queryKey: ['reports'],
    queryFn: () => api.listReports(),
  });
};

/**
 * Fetch specific report
 */
export const useReport = (reportName: string | undefined) => {
  return useQuery({
    queryKey: ['report', reportName],
    queryFn: () => api.getReport(reportName!),
    enabled: !!reportName,
  });
};

/**
 * Fetch log file
 */
export const useLog = (runId: string | undefined) => {
  return useQuery({
    queryKey: ['log', runId],
    queryFn: () => api.getLog(runId!),
    enabled: !!runId,
  });
};
