import { useEffect, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useProgressStore } from './store/progressStore';
import './App.css';

function App() {
  const { updateProgress, currentRun, isLive } = useProgressStore();
  const [showStatus, setShowStatus] = useState(false);

  // Connect to WebSocket for real-time updates
  // Note: WebSocket sends current state immediately on connection,
  // so no separate REST API fetch is needed
  const { connectionState } = useWebSocket({
    onMessage: (data) => {
      updateProgress(data);
    },
  });

  // Simple: show status once connected, hide initially
  useEffect(() => {
    if (connectionState === 'connected') {
      setShowStatus(true);
    }
  }, [connectionState]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>GENML PIPELINE DASHBOARD</h1>
        <div className="header-status">
          {showStatus && (
            <span className="status-indicator connected" role="status" aria-live="polite">
              <span className="status-icon connected" aria-hidden="true" />
              <span className="status-text">Connected</span>
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
                <span className={`status-badge ${showStatus ? 'connected' : 'connecting'}`}>
                  {showStatus ? 'LINK ESTABLISHED' : 'INITIALIZING...'}
                </span>
              </p>
              {showStatus && (
                <p className="success-text">SYSTEM ONLINE - MONITORING ACTIVE</p>
              )}
              {!showStatus && (
                <p className="warning-text">ESTABLISHING CONNECTION...</p>
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
