import { useEffect, useState, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useProgressStore } from './store/progressStore';
import { api } from './api/client';
import type { ConnectionState } from './hooks/useWebSocket';
import './App.css';

const RECONNECTING_GRACE_PERIOD = 4000; // 4 seconds before showing "Reconnecting"
const DISCONNECTED_GRACE_PERIOD = 8000; // 8 seconds total before showing "Disconnected"

function App() {
  const { updateProgress, setConnectionStatus, currentRun, connectionStatus, isLive } = useProgressStore();
  const [displayStatus, setDisplayStatus] = useState<ConnectionState | null>(null); // null = show nothing initially
  const reconnectTimerRef = useRef<number | null>(null);
  const disconnectTimerRef = useRef<number | null>(null);
  const hasEverConnectedRef = useRef(false); // Track if we've ever successfully connected
  const connectionStatusRef = useRef<ConnectionState>(connectionStatus);

  // Fetch initial data on mount (fallback if WebSocket is slow)
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const data = await api.getStatus();
        console.log('Fetched initial status via REST API:', data);
        updateProgress(data);
      } catch (error) {
        console.error('Failed to fetch initial status:', error);
      }
    };

    fetchInitialData();
  }, [updateProgress]);

  // Connect to WebSocket for real-time updates
  const { connectionState, isConnected, lastMessage } = useWebSocket({
    onMessage: (data) => {
      console.log('Received WebSocket update:', data);
      updateProgress(data);
    },
    onStateChange: (state) => {
      console.log('WebSocket state changed:', state);
      setConnectionStatus(state);
    },
  });

  // Handle delayed state changes to avoid flickering
  useEffect(() => {
    // Update ref with current status
    connectionStatusRef.current = connectionStatus;

    // Clear any pending timers
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (disconnectTimerRef.current) {
      clearTimeout(disconnectTimerRef.current);
      disconnectTimerRef.current = null;
    }

    if (connectionStatus === 'connected') {
      // First successful connection or reconnection
      hasEverConnectedRef.current = true;
      setDisplayStatus('connected');
      console.log('[App] Connected - showing Connected status');
    } else if (connectionStatus === 'connecting') {
      // Initial connection attempt - don't show anything until connected
      if (!hasEverConnectedRef.current) {
        console.log('[App] Initial connection attempt - showing nothing');
        setDisplayStatus(null);
      } else {
        // We've connected before, this is a reconnect - keep showing Connected
        console.log('[App] Reconnecting (connecting state) - keeping Connected display');
        setDisplayStatus('connected');
      }
    } else if (connectionStatus === 'reconnecting' || connectionStatus === 'disconnected') {
      // Connection lost - but keep showing "Connected" unless it persists
      if (hasEverConnectedRef.current) {
        console.log('[App] Connection lost - keeping Connected display, starting grace period...');
        setDisplayStatus('connected'); // Keep showing connected

        // After grace period, show reconnecting
        reconnectTimerRef.current = window.setTimeout(() => {
          const currentStatus = connectionStatusRef.current;
          if (currentStatus === 'reconnecting' || currentStatus === 'disconnected') {
            console.log('[App] Grace period expired, showing Reconnecting...');
            setDisplayStatus('reconnecting');

            // After additional time, show disconnected
            disconnectTimerRef.current = window.setTimeout(() => {
              const stillBadStatus = connectionStatusRef.current;
              if (stillBadStatus === 'reconnecting' || stillBadStatus === 'disconnected') {
                console.log('[App] Still offline, showing Disconnected');
                setDisplayStatus('disconnected');
              }
            }, DISCONNECTED_GRACE_PERIOD - RECONNECTING_GRACE_PERIOD);
          }
        }, RECONNECTING_GRACE_PERIOD);
      } else {
        // Never connected, show current state
        setDisplayStatus(connectionStatus);
      }
    } else if (connectionStatus === 'error') {
      // Show error immediately only if we've never connected
      if (!hasEverConnectedRef.current) {
        setDisplayStatus('error');
      } else {
        // We were connected before, treat like disconnection
        setDisplayStatus('connected');
        reconnectTimerRef.current = window.setTimeout(() => {
          setDisplayStatus('error');
        }, RECONNECTING_GRACE_PERIOD);
      }
    }

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (disconnectTimerRef.current) {
        clearTimeout(disconnectTimerRef.current);
      }
    };
  }, [connectionStatus]);

  useEffect(() => {
    if (lastMessage) {
      console.log('Received progress update:', lastMessage);
    }
  }, [lastMessage]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>GENML PIPELINE DASHBOARD</h1>
        <div className="header-status">
          {displayStatus !== null && (
            <span className={`status-indicator ${displayStatus}`} role="status" aria-live="polite">
              <span
                className={`status-icon ${displayStatus}`}
                aria-hidden="true"
              />
              <span className="status-text">
                {displayStatus === 'connected' && 'Connected'}
                {displayStatus === 'connecting' && 'Connecting…'}
                {displayStatus === 'reconnecting' && 'Reconnecting…'}
                {displayStatus === 'disconnected' && 'Disconnected'}
                {displayStatus === 'error' && 'Error'}
              </span>
            </span>
          )}
        </div>
      </header>

      <main className="app-main">
        {(!currentRun || currentRun.status === 'idle' || !currentRun.run_id) && (
          <div className="empty-state">
            <h2>⚠ SYSTEM IDLE ⚠</h2>
            <p>{currentRun?.message || 'AWAITING PIPELINE INITIALIZATION...'}</p>
            <p className="hint">
              INITIATE SEQUENCE: <code>python src/genML/main.py</code>
            </p>
            <div className="connection-info">
              <p>
                <strong>NETWORK STATUS:</strong>{' '}
                <span className={`status-badge ${displayStatus || 'connecting'}`}>
                  {displayStatus === 'connected' ? 'LINK ESTABLISHED' :
                   displayStatus === 'reconnecting' ? 'RECONNECTING...' :
                   displayStatus === 'disconnected' ? '⚠ LINK DOWN' :
                   displayStatus === null ? 'INITIALIZING...' :
                   'INITIALIZING...'}
                </span>
              </p>
              {displayStatus === 'connected' && (
                <p className="success-text">SYSTEM ONLINE - MONITORING ACTIVE</p>
              )}
              {displayStatus === 'reconnecting' && (
                <p className="warning-text">
                  ⚠ CONNECTION INTERRUPTED - ATTEMPTING RECONNECT...
                </p>
              )}
              {displayStatus === 'disconnected' && (
                <p className="warning-text">
                  ⚠ API SERVER OFFLINE - EXECUTE: <code>python run_api.py</code>
                </p>
              )}
              {displayStatus === null && (
                <p className="warning-text">
                  ESTABLISHING CONNECTION...
                </p>
              )}
            </div>
          </div>
        )}

        {currentRun && currentRun.status !== 'idle' && currentRun.run_id && (
          <div className="dashboard">
            {/* Pipeline Overview */}
            <section className="card">
              <h2>PIPELINE STATUS</h2>
              <div className="status-grid">
                <div className="status-item">
                  <span className="label">Run ID:</span>
                  <span className="value">{currentRun.run_id || 'N/A'}</span>
                </div>
                <div className="status-item">
                  <span className="label">Dataset:</span>
                  <span className="value">{currentRun.dataset_name || 'N/A'}</span>
                </div>
                <div className="status-item">
                  <span className="label">Status:</span>
                  <span className={`badge status-${currentRun.status}`}>
                    {currentRun.status.toUpperCase()}
                  </span>
                </div>
                <div className="status-item">
                  <span className="label">Current Stage:</span>
                  <span className="value">
                    {currentRun.current_stage_name || 'N/A'} ({currentRun.current_stage || 0}/5)
                  </span>
                </div>
              </div>
            </section>

            {/* Current Task */}
            {isLive && currentRun.current_task && (
              <section className="card highlight">
                <h2>ACTIVE PROCESS</h2>
                <div className="current-task">
                  <div className="task-description">{currentRun.current_task}</div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${currentRun.stage_progress_pct || 0}%` }}
                    ></div>
                  </div>
                  <div className="progress-text">{currentRun.stage_progress_pct || 0}% complete</div>
                </div>
              </section>
            )}

            {/* Stages */}
            <section className="card">
              <h2>EXECUTION STAGES</h2>
              <div className="stages">
                {Object.entries(currentRun.stages || {}).map(([num, stage]) => (
                  <div key={num} className={`stage stage-${stage.status}`}>
                    <div className="stage-header">
                      <span className="stage-number">{num}</span>
                      <span className="stage-name">{stage.name}</span>
                      <span className={`stage-status status-${stage.status}`}>
                        {stage.status}
                      </span>
                    </div>
                    {stage.duration_seconds !== null && (
                      <div className="stage-duration">
                        Duration: {stage.duration_seconds}s
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* Models */}
            {currentRun.models && currentRun.models.length > 0 && (
              <section className="card">
                <h2>MODEL TRAINING</h2>
                <div className="models-grid">
                  {currentRun.models.map((model) => (
                    <div key={model.name} className={`model-card model-${model.status}`}>
                      <div className="model-name">{model.name}</div>
                      <div className="model-status">{model.status}</div>
                      {model.mean_score !== null && (
                        <div className="model-score">
                          Score: {model.mean_score.toFixed(4)}
                        </div>
                      )}
                      {model.current_trial !== undefined && model.total_trials && (
                        <div className="model-trials">
                          Trial: {model.current_trial}/{model.total_trials}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Resources */}
            {currentRun.resources && (
              <section className="card">
                <h2>SYSTEM RESOURCES</h2>
                <div className="resources-grid">
                  <div className="resource-item">
                    <span className="resource-label">GPU Memory:</span>
                    <span className="resource-value">
                      {currentRun.resources.gpu_memory_mb.toFixed(0)} MB /{' '}
                      {currentRun.resources.gpu_memory_total_mb.toFixed(0)} MB
                    </span>
                  </div>
                  <div className="resource-item">
                    <span className="resource-label">CPU:</span>
                    <span className="resource-value">
                      {currentRun.resources.cpu_percent.toFixed(1)}%
                    </span>
                  </div>
                  <div className="resource-item">
                    <span className="resource-label">RAM:</span>
                    <span className="resource-value">
                      {(currentRun.resources.ram_mb / 1024).toFixed(2)} GB
                    </span>
                  </div>
                  <div className="resource-item">
                    <span className="resource-label">Elapsed Time:</span>
                    <span className="resource-value">
                      {currentRun.resources.elapsed_seconds}s
                    </span>
                  </div>
                </div>
              </section>
            )}

            {/* Raw Data (for debugging) */}
            <details className="card">
              <summary>RAW DATA [DEBUG MODE]</summary>
              <pre className="json-view">{JSON.stringify(currentRun, null, 2)}</pre>
            </details>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
