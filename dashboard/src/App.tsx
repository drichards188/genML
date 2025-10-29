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
            <section className="card glass-overview">
              <div className="glass-overview__header">
                <div>
                  <h2>Pipeline Status</h2>
                  <p className="glass-overview__subtitle">
                    Live telemetry for the current orchestration window
                  </p>
                </div>
                <span className={`pulse-badge status-${currentRun.status}`}>
                  {currentRun.status.toUpperCase()}
                </span>
              </div>
              <div className="overview-grid">
                <div className="overview-item">
                  <span className="overview-label">Run ID</span>
                  <span className="overview-value">{currentRun.run_id || 'N/A'}</span>
                  <span className="overview-caption">Unique execution hash</span>
                </div>
                <div className="overview-item">
                  <span className="overview-label">Dataset</span>
                  <span className="overview-value">{currentRun.dataset_name || 'N/A'}</span>
                  <span className="overview-caption">Active training input</span>
                </div>
                <div className="overview-item">
                  <span className="overview-label">Current Stage</span>
                  <span className="overview-value">
                    {currentRun.current_stage_name || 'N/A'}
                  </span>
                  <span className="overview-caption">
                    Stage {currentRun.current_stage || 0} of 5
                  </span>
                </div>
                <div className="overview-item">
                  <span className="overview-label">Progress</span>
                  <span className="overview-value">
                    {currentRun.stage_progress_pct?.toFixed(0) || 0}%
                  </span>
                  <span className="overview-caption">Stage completion</span>
                </div>
              </div>
            </section>

            <div className="dashboard-grid">
              <div className="dashboard-column primary">
                {isLive && currentRun.current_task && (
                  <section className="card highlight glass">
                    <div className="panel-heading">
                      <h2>Active Process</h2>
                      <span className="panel-pulse">LIVE</span>
                    </div>
                    <div className="current-task">
                      <div className="task-description">{currentRun.current_task}</div>
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{ width: `${currentRun.stage_progress_pct || 0}%` }}
                        ></div>
                      </div>
                      <div className="progress-text">
                        {currentRun.stage_progress_pct || 0}% complete
                      </div>
                    </div>
                  </section>
                )}

                <section className="card glass timeline-card">
                  <div className="panel-heading">
                    <h2>Execution Stages</h2>
                    <span className="panel-caption">
                      Monitors each CrewAI segment with real-time state
                    </span>
                  </div>
                  <div className="timeline">
                    {Object.entries(currentRun.stages || {}).map(([num, stage]) => (
                      <div key={num} className={`timeline-row stage-${stage.status}`}>
                        <div className="timeline-marker">
                          <span className="marker">{num}</span>
                        </div>
                        <div className="timeline-content">
                          <div className="timeline-header">
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
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              <div className="dashboard-column secondary">
                {currentRun.models && currentRun.models.length > 0 && (
                  <section className="card glass models-card">
                    <div className="panel-heading">
                      <h2>Model Training</h2>
                      <span className="panel-caption">
                        Comparative metrics rolling in from experimentation
                      </span>
                    </div>
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

                {currentRun.resources && (
                  <section className="card glass resources-card">
                    <div className="panel-heading">
                      <h2>System Resources</h2>
                      <span className="panel-caption">
                        Telemetry from orchestrator instrumentation
                      </span>
                    </div>
                    <div className="resources-grid">
                      <div className="resource-item">
                        <span className="resource-label">GPU Memory</span>
                        <span className="resource-value">
                          {currentRun.resources.gpu_memory_mb.toFixed(0)} MB /{' '}
                          {currentRun.resources.gpu_memory_total_mb.toFixed(0)} MB
                        </span>
                      </div>
                      <div className="resource-item">
                        <span className="resource-label">CPU Utilization</span>
                        <span className="resource-value">
                          {currentRun.resources.cpu_percent.toFixed(1)}%
                        </span>
                      </div>
                      <div className="resource-item">
                        <span className="resource-label">RAM Footprint</span>
                        <span className="resource-value">
                          {(currentRun.resources.ram_mb / 1024).toFixed(2)} GB
                        </span>
                      </div>
                      <div className="resource-item">
                        <span className="resource-label">Elapsed Time</span>
                        <span className="resource-value">
                          {currentRun.resources.elapsed_seconds}s
                        </span>
                      </div>
                    </div>
                  </section>
                )}
              </div>
            </div>

            <details className="card glass detail-card">
              <summary>Raw Data · Debug Mode</summary>
              <pre className="json-view">{JSON.stringify(currentRun, null, 2)}</pre>
            </details>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
