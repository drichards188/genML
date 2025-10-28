# CrewAI Generic ML Pipeline 🤖

A generalized machine learning pipeline using CrewAI Flows that can work with any dataset following the standard train.csv/test.csv format. Originally designed for the Kaggle Titanic challenge, it has evolved into a flexible ML automation tool.

## Overview

This project demonstrates how to use CrewAI's Flow system to orchestrate a complete machine learning pipeline that automatically adapts to different datasets and problem types:

- **Automated Data Analysis**: Loads and explores any dataset structure
- **Intelligent Feature Engineering**: Uses `AutoFeatureEngine` to automatically detect data types and generate domain-specific features
- **Adaptive Model Training**: Automatically detects problem type (regression vs classification) and trains appropriate models
- **Smart Submission Generation**: Detects submission format from sample files and creates properly formatted outputs

## Project Structure

```
genML/
├── README.md                    # This file
├── CLAUDE.md                    # Development guidance for Claude Code
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Package configuration
├── run_api.py                  # FastAPI server for dashboard backend
├── src/genML/              # Main source code
│   ├── main.py                 # Main entry point
│   ├── flow.py                 # CrewAI Flow orchestration
│   ├── tools.py                # Backwards-compatible shim (re-exports modern pipeline API)
│   ├── submission_formatter.py # Adaptive submission formatting
│   ├── api/                    # Dashboard API backend
│   ├── pipeline/               # Modular ML pipeline implementation
│   │   ├── __init__.py         # Public pipeline API (`load_dataset`, `train_model_pipeline`, ...)
│   │   ├── config.py           # Shared constants, paths, and tuning settings
│   │   ├── dataset.py          # Dataset discovery / loading utilities
│   │   ├── feature_engineering.py # Automated feature engineering + AI ideation integration
│   │   ├── model_advisor.py    # Model guidance + problem type detection
│   │   ├── training.py         # Optuna tuning, ensemble evaluation, GPU-aware training
│   │   ├── prediction.py       # Submission + prediction generation
│   │   └── utils.py            # Memory / GPU cleanup helpers
│   └── features/               # Automated feature engineering helpers
│       ├── feature_engine.py   # Main feature engineering orchestrator
│       ├── data_analyzer.py    # Data type detection
│       ├── feature_processors.py # Feature transformation modules
│       ├── feature_selector.py # Intelligent feature selection
│       └── domain_researcher.py # Domain-specific strategies
├── dashboard/                  # Real-time monitoring dashboard
│   ├── src/                   # React + TypeScript source
│   │   ├── api/               # API client
│   │   ├── hooks/             # React hooks (WebSocket, data fetching)
│   │   ├── store/             # State management
│   │   ├── types/             # TypeScript types
│   │   └── App.tsx            # Main dashboard component
│   ├── package.json           # Node.js dependencies
│   └── vite.config.ts         # Vite configuration
├── datasets/                   # Organized dataset storage
│   ├── current/               # Active dataset (recommended location)
│   ├── titanic/               # Titanic-specific data
│   └── [other]/               # Additional problem domains
├── outputs/                   # Generated artifacts
│   ├── data/                  # Processed datasets
│   ├── features/              # Feature engineering outputs
│   ├── models/                # Trained models
│   ├── predictions/           # Prediction files
│   └── reports/               # Analysis reports
└── submission.csv             # Main submission file (after running)
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # OR using UV (recommended)
   uv sync
   ```

2. **Optional: Install dashboard dependencies**
   ```bash
   # For the real-time monitoring dashboard
   pip install fastapi uvicorn[standard] websockets

   # Frontend dependencies
   cd dashboard
   npm install
   cd ..
   ```

3. **Prepare your dataset:**
   - Place `train.csv` and `test.csv` in `datasets/current/` (recommended)
   - Optionally include `sample_submission.csv` for automatic format detection
   - The pipeline works with any ML dataset following this structure

## Usage

### Quick Start

```bash
# Run the complete ML pipeline
python src/genML/main.py

# Alternative methods
crewai run              # Using CrewAI CLI
uv run kickoff          # Using UV package scripts
```

### What Happens

The pipeline executes in 4 sequential steps using CrewAI Flow orchestration:

1. **Automated Data Analysis** 📊
   - Loads train.csv and test.csv from multiple possible locations
   - Analyzes dataset structure, data types, and missing values
   - Provides comprehensive data summary and statistics

2. **Intelligent Feature Engineering** 🔧
   - **AutoFeatureEngine** automatically detects data types (numerical, categorical, text, datetime)
   - Generates domain-specific features based on detected patterns
   - Applies appropriate preprocessing (scaling, encoding, text processing)
   - Performs intelligent feature selection using statistical and model-based methods
   - Adapts to any dataset structure automatically

3. **Adaptive Model Training** 🤖
   - **Automatic problem type detection** (regression vs classification)
   - Trains appropriate models based on problem type:
     - **Classification**: Logistic Regression, Random Forest, XGBoost Classifier
     - **Regression**: Linear Regression, Random Forest Regressor, XGBoost Regressor
   - Uses proper cross-validation (StratifiedKFold for classification, KFold for regression)
   - Selects best performing model using appropriate metrics

4. **Smart Submission Generation** 📈
   - **Adaptive submission formatting** detects format from sample_submission.csv
   - Supports binary classification, probability outputs, and regression
   - Generates predictions with confidence scores
   - Creates properly formatted submission files
   - Works with any competition platform format

## Output Files

After successful execution, you'll find:

**Main Files:**
- `submission.csv` - Main submission file ready for upload
- `submission_YYYYMMDD_HHMMSS.csv` - Timestamped backup

**Organized Outputs Directory:**
- `outputs/data/` - Processed datasets (train_data.pkl, test_data.pkl)
- `outputs/features/` - Feature engineering artifacts (X_train.npy, feature_names.txt)
- `outputs/models/` - Trained models (best_model_*.pkl, cross_validation_results.csv)
- `outputs/predictions/` - Detailed predictions and submission files
- `outputs/reports/` - JSON reports from each pipeline stage

## Real-Time Dashboard 📊

The project includes a **real-time monitoring dashboard** built with React + TypeScript that provides live updates of pipeline execution.

### Dashboard Features

- **Live Pipeline Status**: Real-time progress tracking via WebSocket
- **Stage Monitoring**: Visual indicators for all 5 pipeline stages (pending/running/completed/failed)
- **Model Training Progress**: Track each model's training status and Optuna trials
- **Resource Monitoring**: GPU memory, CPU, and RAM usage
- **Current Activity**: See exactly what the pipeline is doing right now
- **Debug View**: Raw JSON data explorer for development

### Quick Start - Dashboard

**Prerequisites:**
- Node.js 18+ and npm
- FastAPI dependencies: `pip install fastapi uvicorn[standard] websockets`

**Development Mode (Recommended):**

```bash
# Terminal 1: Start API Backend
python run_api.py

# Terminal 2: Start Dashboard (in new terminal)
cd dashboard
npm install  # First time only
npm run dev

# Terminal 3: Run the ML Pipeline (in new terminal)
python src/genML/main.py
```

Then open your browser to **http://localhost:5173** to watch the pipeline run in real-time!

**Production Mode:**

```bash
# Build and serve the dashboard
cd dashboard
npm run build
cd ..
python run_api.py

# Open browser to http://localhost:8000
```

### Dashboard Architecture

- **Frontend**: React 18 + TypeScript + Vite
- **Backend**: FastAPI with WebSocket support
- **State Management**: Zustand
- **Charts**: Recharts (for future visualizations)
- **API Client**: Axios with React Query

The dashboard auto-connects to the backend and displays real-time updates as the pipeline executes. When no pipeline is running, it shows "No Active Pipeline" status.

**For more details**, see `dashboard/README.md`


## Key Features

### 🤖 Automated Feature Engineering
- **AutoFeatureEngine**: Automatically detects data types and generates relevant features
- **Domain-specific strategies**: Adapts feature generation based on detected problem domains
- **Intelligent feature selection**: Uses statistical tests and model-based selection
- **Configurable processing**: Customizable feature engineering parameters

### 🔄 Adaptive Pipeline
- **Problem type detection**: Automatically identifies regression vs classification
- **Multi-dataset support**: Works with any train.csv/test.csv dataset
- **Smart path resolution**: Finds datasets in multiple organized locations
- **Format detection**: Automatically detects submission format from sample files

### 🏗️ CrewAI Flow Architecture
- **Sequential orchestration**: Uses `@start()` and `@listen()` decorators
- **Error handling**: Comprehensive error propagation between pipeline stages
- **Progress tracking**: Real-time status updates and detailed logging
- **Modular design**: Each stage is independent and can be modified separately
- **Pipeline package**: All production logic lives in `src/genML/pipeline/`, while `src/genML/tools.py` keeps the historical API intact for older scripts/tests.

### 📊 Core ML Functions
The primary entry points are re-exported via `src.genML.pipeline` (and mirrored in `src.genML.tools` for compatibility):
- `load_dataset()` - Multi-path dataset loading with validation
- `engineer_features()` - Automated feature engineering pipeline (with optional AI feature ideation)
- `train_model_pipeline()` - Multi-model training with Optuna tuning, GPU-aware factories, and optional stacking ensembles
- `generate_predictions()` - Adaptive prediction generation and formatting

## Requirements

**Core Dependencies:**
- Python 3.10+ (as specified in pyproject.toml)
- CrewAI 0.193.2+ with tools support
- Standard ML libraries (pandas, scikit-learn, xgboost, numpy)
- Any dataset following train.csv/test.csv format

**Optional - For Real-Time Dashboard:**
- Node.js 18+ and npm
- FastAPI + uvicorn + websockets
- Modern web browser

## Working with Different Datasets

To use the pipeline with a new dataset:

1. **Prepare your data:**
   - Place `train.csv` and `test.csv` in `datasets/current/`
   - Optionally include `sample_submission.csv` for format detection
   - Ensure your train.csv has the target variable in a column

2. **Run the pipeline:**
   ```bash
   conda source genml
   python3 src/genML/main.py
   ```

3. **The pipeline will automatically:**
   - Detect whether it's a regression or classification problem
   - Generate appropriate features based on your data types
   - Train suitable models for your problem type
   - Create properly formatted submission files

## Troubleshooting

**Missing data files:**
```
❌ Missing required data files: train.csv, test.csv
```
Solution: Place your dataset files in `datasets/current/` or project root.

**Import errors:**
```
ModuleNotFoundError: No module named 'crewai'
```
Solution: `pip install -r requirements.txt` or `uv sync`

**Pipeline fails on feature engineering:**
Check that your dataset has proper column names and no completely empty columns.

**Dashboard not connecting:**
- Ensure FastAPI backend is running: `python run_api.py`
- Check that port 8000 is available
- Verify dashboard is running: `cd dashboard && npm run dev`
- See `dashboard/README.md` for detailed troubleshooting
