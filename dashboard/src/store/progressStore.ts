/**
 * Zustand store for managing pipeline progress state.
 * Centralized state management for the entire dashboard.
 */

import { create } from 'zustand';
import type { RunProgress } from '../types/pipeline';
import type { ConnectionState } from '../hooks/useWebSocket';

export type ViewMode = 'live' | 'post-run' | 'history';

interface ProgressState {
  // Current run progress data
  currentRun: RunProgress | null;

  // Connection status
  connectionStatus: ConnectionState;
  isLive: boolean;

  // View mode
  viewMode: ViewMode;

  // Selected run for history view
  selectedRunId: string | null;

  // Actions
  updateProgress: (data: RunProgress) => void;
  setConnectionStatus: (status: ConnectionState) => void;
  setViewMode: (mode: ViewMode) => void;
  setSelectedRunId: (runId: string | null) => void;
  reset: () => void;
}

const initialState = {
  currentRun: null,
  connectionStatus: 'disconnected' as ConnectionState,
  isLive: false,
  viewMode: 'live' as ViewMode,
  selectedRunId: null,
};

export const useProgressStore = create<ProgressState>((set) => ({
  ...initialState,

  updateProgress: (data: RunProgress) =>
    set((state) => {
      const isLive = data.status === 'running';

      // Auto-switch to post-run view when pipeline completes
      const viewMode =
        state.viewMode === 'live' && data.status === 'completed'
          ? 'post-run'
          : state.viewMode;

      return {
        currentRun: data,
        isLive,
        viewMode,
      };
    }),

  setConnectionStatus: (status: ConnectionState) =>
    set({ connectionStatus: status }),

  setViewMode: (mode: ViewMode) =>
    set({ viewMode: mode }),

  setSelectedRunId: (runId: string | null) =>
    set({ selectedRunId: runId }),

  reset: () => set(initialState),
}));

export default useProgressStore;
