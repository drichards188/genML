# GenML Pipeline Dashboard

Real-time monitoring dashboard for the GenML machine learning pipeline.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+ with pipeline dependencies installed
- FastAPI backend running

### Installation

```bash
# From the dashboard directory
cd dashboard
npm install
```

### Running the Dashboard

**Option 1: Development Mode (Recommended)**

Terminal 1 - Start FastAPI backend:
```bash
# From project root
pip install -r requirements-dashboard.txt
python run_api.py
```

Terminal 2 - Start React dev server:
```bash
# From dashboard directory
cd dashboard
npm run dev
```

Terminal 3 - Run the ML pipeline:
```bash
# From project root
python src/genML/main.py
```

Then open your browser to: **http://localhost:5173**

**Option 2: Production Mode**

```bash
# Build the React app
cd dashboard
npm run build

# Start API server (serves built React app)
cd ..
python run_api.py

# Open browser to http://localhost:8000
```

## 📊 Features

### Current Features (v1.0 - Minimal Working Dashboard)

✅ **Real-time WebSocket Connection**
- Auto-connect/reconnect to backend
- Live status indicator
- Real-time progress updates

✅ **Pipeline Status Overview**
- Run ID, dataset name, status
- Current stage and progress
- Elapsed time

✅ **Current Activity Monitoring**
- Current task description
- Progress bar with percentage
- Live updates

✅ **Stage Tracking**
- All 5 pipeline stages
- Status indicators (pending/running/completed/failed)
- Duration for completed stages

✅ **Model Training Progress**
- List of all models being trained
- Training status for each model
- Cross-validation scores
- Optuna trial progress (trial X/Y)

✅ **Resource Monitoring**
- GPU memory usage
- CPU utilization
- RAM consumption
- Elapsed time

✅ **Debug View**
- Raw JSON data explorer
- Useful for development and debugging

### Planned Features (Future Enhancements)

⏳ **Advanced Visualizations**
- Real-time charts for resource usage (Recharts)
- Model performance comparison charts
- Optuna optimization history plots

⏳ **Post-Run Analysis**
- Detailed model comparison
- Feature importance visualization
- AI error pattern analysis
- Timeline Gantt chart

⏳ **History & Reports**
- Run history browser
- Report file viewer
- Log file viewer
- Downloadable reports

⏳ **Enhanced UI**
- Dark/light mode toggle
- Customizable dashboard layout
- Keyboard shortcuts
- Search and filtering

## 🛠️ Development

### Project Structure

```
dashboard/
├── src/
│   ├── api/              # API client for backend
│   ├── hooks/            # React hooks (WebSocket, data fetching)
│   ├── store/            # Zustand state management
│   ├── types/            # TypeScript type definitions
│   ├── components/       # React components (to be expanded)
│   ├── App.tsx           # Main application component
│   ├── App.css          # Styling
│   └── main.tsx          # Entry point
├── package.json
├── vite.config.ts        # Vite configuration
└── tsconfig.json         # TypeScript configuration
```

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### Adding New Components

1. Create component in `src/components/` directory
2. Import and use in `App.tsx` or other components
3. Add TypeScript types in `src/types/pipeline.ts` if needed
4. Update this README

## 🔌 API Integration

The dashboard connects to the FastAPI backend via:

- **REST API**: HTTP requests for static data (reports, logs, etc.)
- **WebSocket**: Real-time progress updates at `ws://localhost:8000/ws/progress`

API endpoints are automatically proxied in development mode (configured in `vite.config.ts`).

### Available API Endpoints

- `GET /api/health` - Health check
- `GET /api/status` - Current pipeline status
- `GET /api/runs` - List all runs
- `GET /api/runs/{run_id}` - Get specific run
- `GET /api/reports` - List reports
- `GET /api/reports/{name}` - Get report
- `GET /api/logs/{run_id}` - Get log file
- `GET /api/models` - Model comparison data
- `WS /ws/progress` - Real-time progress stream

## 🐛 Troubleshooting

### WebSocket Not Connecting

- Ensure FastAPI server is running (`python run_api.py`)
- Check that port 8000 is not in use
- Verify WebSocket URL in browser console

### Dashboard Shows "No Active Pipeline"

- This is normal when no pipeline is running
- Start the pipeline with `python src/genML/main.py`
- Dashboard will automatically detect and display progress

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

```bash
# Change port in vite.config.ts
server: {
  port: 3000,  # Change to different port
  ...
}
```

## 📝 Notes

- This is v1.0 - A minimal working dashboard
- More components and features will be added incrementally
- See `DASHBOARD_IMPLEMENTATION.md` in project root for full roadmap
- The dashboard is designed to be expanded - feel free to add components!

## 🤝 Contributing

To add new features:

1. Create components in `src/components/`
2. Add data fetching hooks in `src/hooks/`
3. Update types in `src/types/pipeline.ts`
4. Test with real pipeline execution
5. Update this README

## 📄 License

Part of the GenML project.
