# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CrewAI ML Pipeline** project that demonstrates automated machine learning workflows using CrewAI Flows. The system is designed to work with any dataset following the standard train.csv/test.csv format, making it a generalized ML pipeline tool.

## Key Commands

### Running the Pipeline
```bash
# Main execution (recommended)
python src/genML/main.py

# Alternative using CrewAI CLI
crewai run

# Using UV package scripts
uv run kickoff
uv run run_pipeline
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt
# OR using UV
uv sync

# View flow diagram
python -c "from src.genML.main import plot; plot()"
# OR
uv run plot
```

### Testing
The project currently has no formal test suite. Test files would go in the `tests/` directory. To test the pipeline, use it with actual datasets.

## Architecture

### Core Flow Structure
The system uses **CrewAI Flows** with a 4-stage sequential pipeline:

1. **Data Loading** (`load_data_task`) - Loads and validates datasets
2. **Feature Engineering** (`feature_engineering_task`) - Automated feature generation using `AutoFeatureEngine`
3. **Model Training** (`model_training_task`) - Multi-model training with cross-validation
4. **Prediction Generation** (`prediction_task`) - Generates final submissions

Each stage is decorated with `@start()` or `@listen()` and includes comprehensive error handling.

### Key Components

**Flow Orchestration** (`src/genML/flow.py`):
- `MLPipelineFlow` class manages the entire pipeline
- Uses CrewAI's `@start()` and `@listen()` decorators for sequencing
- Each stage returns JSON results for the next stage

**Core ML Functions** (`src/genML/tools.py`):
- `load_dataset()` - Dataset loading with multiple path resolution
- `engineer_features()` - Automated feature engineering pipeline
- `train_model_pipeline()` - Model training with automatic problem type detection
- `generate_predictions()` - Prediction generation with adaptive submission formatting

**Automated Feature Engineering** (`src/genML/features/`):
- `AutoFeatureEngine` - Main feature engineering orchestrator
- `DataTypeAnalyzer` - Automatic data type detection
- Feature processors for numerical, categorical, text, and datetime data
- `AdvancedFeatureSelector` - Intelligent feature selection
- `DomainResearcher` - Domain-specific feature strategies

**Adaptive Submission Formatting** (`src/genML/submission_formatter.py`):
- Automatically detects submission format from sample files
- Supports binary classification, probability outputs, and regression
- Creates properly formatted submission files

### Directory Structure

**Datasets** - Organized by problem type:
```
datasets/
├── current/     # Active dataset (primary location)
├── titanic/     # Titanic-specific data
├── music/       # Music-related problem data
└── [other]/     # Additional problem domains
```

**Outputs** - Generated artifacts:
```
outputs/
├── data/        # Processed datasets (.pkl files)
├── features/    # Feature engineering artifacts
├── models/      # Trained models and CV results
├── predictions/ # Prediction files and detailed results
└── reports/     # JSON reports from each stage
```

## Problem Type Detection

The system automatically detects whether a problem is regression or classification by analyzing the target variable:
- **Classification**: ≤20 unique values, integer-like values, low unique ratio
- **Regression**: Many unique values, continuous values, high unique ratio

This drives automatic model selection and evaluation metrics.

## Dataset Path Resolution

The system searches for `train.csv` and `test.csv` in this order:
1. `datasets/current/` (recommended)
2. `datasets/active/`
3. Project root directory

## Working with Different Datasets

To work with a new dataset:
1. Place `train.csv` and `test.csv` in `datasets/current/`
2. Optionally include `sample_submission.csv` for format detection
3. Run the pipeline - it will automatically adapt to the new data structure

## Feature Engineering

The `AutoFeatureEngine` provides:
- **Automatic data type detection** (numerical, categorical, text, datetime)
- **Domain-specific strategies** based on detected patterns
- **Configurable processing** via the config dictionary in `engineer_features()`
- **Intelligent feature selection** using statistical and model-based methods

## Model Training

The pipeline trains multiple models automatically:
- **Classification**: Logistic Regression, Random Forest, XGBoost
- **Regression**: Linear Regression, Random Forest Regressor, XGBoost Regressor
- Uses appropriate cross-validation (StratifiedKFold vs KFold)
- Selects best model based on accuracy (classification) or MSE (regression)

## Output Files

After successful execution:
- `submission.csv` - Main submission file for competitions
- `outputs/models/best_model_*.pkl` - Trained model
- `outputs/features/feature_names.txt` - List of engineered features
- `outputs/reports/*.json` - Detailed reports from each stage

## Error Handling

Each pipeline stage includes comprehensive error handling:
- Failed stages return `{"status": "failed", "error": "..."}`
- Downstream stages check for failures and skip execution
- All errors are logged with descriptive messages

## Extending the System

**Adding New Feature Processors**: Create classes in `src/genML/features/feature_processors.py`

**Adding New Models**: Modify the `models` dictionary in `train_model_pipeline()`

**Adding New Domains**: Extend `DomainResearcher` with domain-specific patterns

**Custom Submission Formats**: Extend `SubmissionFormatter` for new competition platforms