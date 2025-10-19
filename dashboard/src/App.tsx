import { useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useProgressStore } from './store/progressStore';
import { api } from './api/client';
import './App.css';

function App() {
  const { updateProgress, setConnectionStatus, currentRun, connectionStatus, isLive } = useProgressStore();

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
  const { isConnected, lastMessage } = useWebSocket({
    onMessage: (data) => {
      console.log('Received WebSocket update:', data);
      updateProgress(data);
    },
    onConnect: () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    },
    onDisconnect: () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
    },
    onError: () => {
      console.log('WebSocket error');
      setConnectionStatus('reconnecting');
    },
  });

  useEffect(() => {
    if (lastMessage) {
      console.log('Received progress update:', lastMessage);
    }
  }, [lastMessage]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 GenML Pipeline Dashboard</h1>
        <div className="header-status">
          <span className={`status-indicator ${connectionStatus}`}>
            {isConnected ? '● Connected' : '○ Disconnected'}
          </span>
        </div>
      </header>

      <main className="app-main">
        {(!currentRun || currentRun.status === 'idle' || !currentRun.run_id) && (
          <div className="empty-state">
            <h2>💤 No Active Pipeline</h2>
            <p>{currentRun?.message || 'Waiting for pipeline to start...'}</p>
            <p className="hint">
              To start the pipeline, run: <code>python src/genML/main.py</code>
            </p>
            <div className="connection-info">
              <p>
                <strong>Connection Status:</strong>{' '}
                <span className={`status-badge ${connectionStatus}`}>
                  {connectionStatus === 'connected' ? '✓ Connected to API' : '⚠ Disconnected'}
                </span>
              </p>
              {connectionStatus === 'connected' && (
                <p className="success-text">✓ Dashboard is ready and waiting for pipeline data</p>
              )}
              {connectionStatus !== 'connected' && (
                <p className="warning-text">
                  ⚠ Cannot connect to API server. Make sure it's running: <code>python run_api.py</code>
                </p>
              )}
            </div>
          </div>
        )}

        {currentRun && currentRun.status !== 'idle' && currentRun.run_id && (
          <div className="dashboard">
            {/* Pipeline Overview */}
            <section className="card">
              <h2>Pipeline Status</h2>
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
                <h2>Current Activity</h2>
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
              <h2>Pipeline Stages</h2>
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
                <h2>Models Training</h2>
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
                <h2>Resource Usage</h2>
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
              <summary>Raw Data (Debug)</summary>
              <pre className="json-view">{JSON.stringify(currentRun, null, 2)}</pre>
            </details>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
