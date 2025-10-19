# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **CrewAI ML Pipeline** project that demonstrates automated machine learning workflows using CrewAI Flows. The system is designed to work with any dataset following the standard train.csv/test.csv format, making it a generalized ML pipeline tool with AI-powered intelligent advisors for automated improvement suggestions.

## Key Commands

### Running the Pipeline
```bash
# Main execution (recommended for GPU support)
conda activate genml
python src/genML/main.py

# Alternative using CrewAI CLI (CPU only)
crewai run

# Using UV package scripts (CPU only)
uv run kickoff
uv run run_pipeline
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Install with GPU support (recommended)
mamba env create -f environment.yml
conda activate genml

# OR using UV (CPU only)
uv sync

# View flow diagram
python -c "from src.genML.main import plot; plot()"
# OR
uv run plot
```

### Testing
```bash
# Run full test suite
pytest

# Run with coverage report
pytest --cov=src/genML --cov-report=term-missing

# Run specific test modules
pytest tests/test_pipeline_modules.py
pytest -k feature_engine

# Test AI advisors separately
export OPENAI_API_KEY='your-api-key'
python test_ai_advisors.py
```

## Architecture

### Core Flow Structure
The system uses **CrewAI Flows** with a 4-stage sequential pipeline:

1. **Data Loading** (`load_data_task`) - Loads and validates datasets
2. **Feature Engineering** (`feature_engineering_task`) - Automated feature generation with AI ideation
3. **Model Training** (`model_training_task`) - Multi-model training with Optuna tuning and ensembles
4. **Prediction Generation** (`prediction_task`) - Generates final submissions

Each stage is decorated with `@start()` or `@listen()` and includes comprehensive error handling.

### Modular Pipeline Architecture

The codebase has been refactored into a modular structure with `src/genML/pipeline/` containing all production logic:

**Flow Orchestration** (`src/genML/flow.py`):
- `MLPipelineFlow` class manages the entire pipeline
- Uses CrewAI's `@start()` and `@listen()` decorators for sequencing
- Each stage returns JSON results for the next stage

**Pipeline Package** (`src/genML/pipeline/`):
- `__init__.py` - Public API exports (load_dataset, engineer_features, train_model_pipeline, generate_predictions)
- `config.py` - Shared constants, paths, tuning settings, and hyperparameter ranges
- `dataset.py` - Dataset discovery and loading utilities with multi-path resolution
- `feature_engineering.py` - Automated feature engineering with AI ideation integration
- `model_advisor.py` - Model selection guidance and problem type detection
- `training.py` - Optuna tuning, ensemble evaluation, GPU-aware training
- `prediction.py` - Submission and prediction generation
- `ai_tuning.py` - AI-powered hyperparameter tuning utilities
- `tuning.py` - Optuna optimization strategies
- `utils.py` - Memory management and GPU cleanup helpers
- `optional_dependencies.py` - Runtime dependency checks for optional features

**Backward Compatibility** (`src/genML/tools.py`):
- Re-exports pipeline functions for legacy compatibility
- Maintains historical API for older scripts and tests
- All new code should import from `src.genML.pipeline` directly

**Automated Feature Engineering** (`src/genML/features/`):
- `feature_engine.py` - Main AutoFeatureEngine orchestrator
- `data_analyzer.py` - Automatic data type detection (DataTypeAnalyzer)
- `feature_processors.py` - Processors for numerical, categorical, text, and datetime data
- `feature_selector.py` - AdvancedFeatureSelector with statistical and model-based selection
- `domain_researcher.py` - DomainResearcher for domain-specific feature strategies

**AI-Powered Advisors** (`src/genML/ai_advisors/`):
- `openai_client.py` - Robust OpenAI API client with response caching and cost tracking
- `error_analyzer.py` - ErrorPatternAnalyzer that identifies prediction error patterns
- `feature_ideation.py` - FeatureIdeationAdvisor that suggests domain-specific features
- `model_selector.py` - AI-powered model selection guidance
- See `AI_ADVISORS_README.md` for comprehensive documentation

**Utilities**:
- `submission_formatter.py` - Adaptive submission formatting for competition platforms
- `logging_config.py` - Centralized logging configuration
- `gpu_utils.py` - GPU detection and memory management utilities

### AI Advisors Integration

The pipeline includes AI-powered intelligent advisors using OpenAI's GPT-4o-mini:

**Feature Ideation Advisor** (Priority 2):
- Runs during feature engineering stage
- Suggests domain-specific features, interactions, and transformations
- Analyzes current features and provides Python formulas for new features
- Expected improvement: 3-12%
- Cost: ~$0.02-0.05 per run

**Error Pattern Analyzer** (Priority 1):
- Runs during model training after best model is trained
- Analyzes prediction errors to identify patterns and correlations
- Suggests targeted improvements for specific subpopulations
- Expected improvement: 5-15%
- Cost: ~$0.01-0.03 per run

**Model Selection Advisor**:
- Provides guidance on which models to prioritize
- Filters out unsuitable models based on dataset characteristics
- Recommends optimal model ordering for training

**Configuration** (in `src/genML/pipeline/config.py`):
```python
AI_ADVISORS_CONFIG = {
    'enabled': True,  # Master switch
    'feature_ideation': {
        'enabled': True,
        'sample_size': 100,
        'save_report': True,
    },
    'error_analysis': {
        'enabled': True,
        'top_n_errors': 100,
        'save_report': True,
    },
    'openai_config': {
        'model': 'gpt-4o-mini',
        'max_cost_per_run': 10.0,
        'enable_cache': True,
        'cache_dir': 'outputs/ai_cache',
    }
}
```

**Cost Management**:
- Response caching saves 80%+ on repeated requests
- Configurable spending caps (default: $10/run)
- Real-time token usage tracking
- Expected total cost: $0.03-0.08 per pipeline run

**Output Reports**:
- `outputs/reports/ai_feature_suggestions.json` - Feature ideas with formulas
- `outputs/reports/ai_error_analysis.json` - Error patterns and priority actions
- `outputs/ai_cache/` - Cached API responses

### Directory Structure

**Source Code**:
```
src/genML/
├── main.py              # Main entry point
├── flow.py              # CrewAI Flow orchestration
├── tools.py             # Legacy API (re-exports pipeline)
├── submission_formatter.py
├── logging_config.py
├── gpu_utils.py
├── pipeline/            # Modular ML pipeline (NEW)
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── feature_engineering.py
│   ├── model_advisor.py
│   ├── training.py
│   ├── prediction.py
│   ├── ai_tuning.py
│   ├── tuning.py
│   └── utils.py
├── ai_advisors/         # AI-powered advisors (NEW)
│   ├── __init__.py
│   ├── openai_client.py
│   ├── error_analyzer.py
│   ├── feature_ideation.py
│   └── model_selector.py
└── features/            # Feature engineering modules
    ├── __init__.py
    ├── feature_engine.py
    ├── data_analyzer.py
    ├── feature_processors.py
    ├── feature_selector.py
    └── domain_researcher.py
```

**Datasets** - Organized by problem type:
```
datasets/
├── current/     # Active dataset (primary location)
├── active/      # Alternative location
├── titanic/     # Titanic-specific data
├── music/       # Music-related problem data
└── [other]/     # Additional problem domains
```

**Outputs** - Generated artifacts:
```
outputs/
├── data/        # Processed datasets (.pkl files)
├── features/    # Feature engineering artifacts (.npy, .txt)
├── models/      # Trained models and CV results
├── predictions/ # Prediction files and detailed results
├── reports/     # JSON reports from each stage
├── ai_cache/    # Cached AI advisor responses (NEW)
└── logs/        # Pipeline execution logs
```

**Testing**:
```
tests/
├── conftest.py               # Shared pytest fixtures
└── test_pipeline_modules.py  # Pipeline module tests (NEW)
```

**Documentation**:
```
├── README.md                       # User-facing documentation
├── CLAUDE.md                       # This file (AI assistant guidance)
├── AGENTS.md                       # Repository coding guidelines (NEW)
├── AI_ADVISORS_README.md           # AI advisors comprehensive guide (NEW)
├── AI_IMPLEMENTATION_SUMMARY.md    # AI implementation details (NEW)
└── SETUP.md                        # Setup instructions
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

## GPU Support

The pipeline includes GPU acceleration support:

**GPU Detection** (`gpu_utils.py`):
- Automatic detection of CUDA, RAPIDS (cuML/cuDF)
- Graceful fallback to CPU if GPU unavailable
- Memory management and cleanup utilities

**GPU-Aware Training** (`pipeline/training.py`):
- GPU model factories for XGBoost, CatBoost, LightGBM
- Automatic device selection based on availability
- Memory optimization for large datasets

**GPU Environment Setup**:
```bash
# Install with GPU support (recommended)
mamba env create -f environment.yml
conda activate genml

# GPU libraries included:
# - cuML/cuDF (RAPIDS) for GPU-accelerated scikit-learn
# - XGBoost with CUDA support
# - CatBoost with GPU training
# - LightGBM with GPU support
```

**Note**: Avoid using `crewai run` with the GPU conda environment as it spawns a uv environment without RAPIDS support. Use `python src/genML/main.py` instead.

## Working with Different Datasets

To work with a new dataset:
1. Place `train.csv` and `test.csv` in `datasets/current/`
2. Optionally include `sample_submission.csv` for format detection
3. Optionally set `OPENAI_API_KEY` environment variable for AI advisors
4. Run the pipeline - it will automatically adapt to the new data structure

## Feature Engineering

The `AutoFeatureEngine` provides:
- **Automatic data type detection** (numerical, categorical, text, datetime)
- **Domain-specific strategies** based on detected patterns
- **Configurable processing** via the config dictionary in `engineer_features()`
- **Intelligent feature selection** using statistical and model-based methods
- **AI-powered feature ideation** (optional, requires OpenAI API key)

## Model Training

The pipeline trains multiple models automatically:

**Classification Models**:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (GPU-accelerated if available)
- CatBoost Classifier (GPU-accelerated if available)
- LightGBM Classifier (GPU-accelerated if available)
- Optional: Stacking Ensemble

**Regression Models**:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (GPU-accelerated if available)
- CatBoost Regressor (GPU-accelerated if available)
- LightGBM Regressor (GPU-accelerated if available)
- Optional: Stacking Ensemble

**Training Features**:
- Optuna hyperparameter tuning with configurable trials
- AI-powered hyperparameter suggestions (optional)
- Cross-validation (StratifiedKFold for classification, KFold for regression)
- Automatic best model selection
- GPU acceleration when available
- Memory-efficient processing

## Output Files

After successful execution:
- `submission.csv` - Main submission file for competitions
- `submission_YYYYMMDD_HHMMSS.csv` - Timestamped backup
- `outputs/models/best_model_*.pkl` - Trained model
- `outputs/models/cross_validation_results.csv` - Detailed CV results
- `outputs/features/feature_names.txt` - List of engineered features
- `outputs/reports/*.json` - Detailed reports from each stage
- `outputs/reports/ai_*.json` - AI advisor suggestions (if enabled)

## Error Handling

Each pipeline stage includes comprehensive error handling:
- Failed stages return `{"status": "failed", "error": "..."}`
- Downstream stages check for failures and skip execution
- All errors are logged with descriptive messages
- AI advisors gracefully degrade if OpenAI API unavailable
- Pipeline continues even if optional features fail

## Testing

The project includes comprehensive test coverage:

**Test Suite** (`tests/test_pipeline_modules.py`):
- Dataset loading and validation tests
- Problem type detection tests
- Feature engineering helper tests
- AI tuning utilities tests
- Model selection guidance tests
- Pipeline stage failure handling tests
- Legacy compatibility tests for `tools.py` re-exports

**Fixtures** (`tests/conftest.py`):
- Sample train/test DataFrames
- Temporary dataset directories
- Pipeline output directory mocking

**Running Tests**:
```bash
# Full test suite
pytest

# With coverage
pytest --cov=src/genML --cov-report=term-missing

# Specific tests
pytest tests/test_pipeline_modules.py -v
pytest -k "test_load_dataset"

# Slow/GPU tests (if marked)
pytest -m slow
```

## Extending the System

**Adding New Feature Processors**:
Create classes in `src/genML/features/feature_processors.py`

**Adding New Models**:
Modify the model factories in `src/genML/pipeline/training.py`:
- Add to `_get_classification_models()` or `_get_regression_models()`
- Add GPU factory if GPU acceleration is available
- Update hyperparameter ranges in `config.py`

**Adding New AI Advisors**:
1. Create new advisor class in `src/genML/ai_advisors/`
2. Follow the pattern of existing advisors (use OpenAIClient)
3. Add configuration to `AI_ADVISORS_CONFIG` in `config.py`
4. Integrate into appropriate pipeline stage in `pipeline/` modules
5. Add comprehensive error handling
6. Update `AI_ADVISORS_README.md`

**Adding New Domains**:
Extend `DomainResearcher` in `src/genML/features/domain_researcher.py` with domain-specific patterns

**Custom Submission Formats**:
Extend `SubmissionFormatter` in `src/genML/submission_formatter.py` for new competition platforms

**Adding New Pipeline Stages**:
1. Create new function in appropriate `pipeline/` module
2. Export from `pipeline/__init__.py`
3. Add `@listen()` decorated method to `MLPipelineFlow` in `flow.py`
4. Update stage sequencing and error handling

## Code Organization Guidelines

See `AGENTS.md` for detailed repository guidelines including:
- Module organization conventions
- Build, test, and development commands
- Coding style and naming conventions (black, isort, flake8, mypy)
- Testing guidelines and fixture usage
- Commit and pull request guidelines
- Environment and data management tips

**Key Principles**:
- All production logic lives in `src/genML/pipeline/`
- `src/genML/tools.py` maintains backward compatibility
- New imports should use `from src.genML.pipeline import ...`
- Use 4-space indentation, snake_case, and descriptive names
- Target Python 3.10+
- Format with `black` and `isort` before committing

## AI Advisors Usage

**Quick Start**:
```bash
# 1. Install OpenAI package (if not already installed)
pip install openai

# 2. Set API key
export OPENAI_API_KEY='your-api-key-here'

# 3. Run pipeline normally
python src/genML/main.py
```

**Testing AI Advisors**:
```bash
# Test independently before full pipeline run
export OPENAI_API_KEY='your-key'
python test_ai_advisors.py
```

**Disabling AI Advisors**:
```python
# In src/genML/pipeline/config.py
AI_ADVISORS_CONFIG = {
    'enabled': False,  # Master switch
    # ... rest of config
}
```

**Cost Management**:
- Default max: $10/run
- Response caching enabled by default
- Expected cost: $0.03-0.08 per run
- Cache persists between runs in `outputs/ai_cache/`

**See Also**:
- `AI_ADVISORS_README.md` - Comprehensive AI advisors documentation
- `AI_IMPLEMENTATION_SUMMARY.md` - Implementation details and architecture decisions

## Performance Optimization

**GPU Acceleration**:
- Use conda environment for GPU support: `conda activate genml`
- GPU detection happens automatically
- Models will use GPU if available and fall back to CPU gracefully

**Memory Management**:
- Pipeline includes automatic memory cleanup between stages
- Large datasets are processed in chunks where possible
- Use `utils.py` cleanup functions for explicit memory management

**Caching**:
- AI advisor responses are cached to save costs and time
- Processed datasets are pickled for faster reloads
- Feature engineered data saved as numpy arrays for efficiency

## Common Workflows

**New Dataset Development**:
1. Place data in `datasets/current/`
2. Run: `python src/genML/main.py`
3. Review AI suggestions in `outputs/reports/ai_*.json`
4. Implement top suggestions in feature engine
5. Re-run pipeline to measure improvement
6. Iterate based on new AI suggestions

**Model Tuning**:
1. Adjust Optuna trials in `pipeline/config.py` (`OPTUNA_N_TRIALS`)
2. Enable AI tuning for hyperparameter suggestions
3. Modify hyperparameter ranges in `config.py`
4. Enable stacking ensemble for best performance

**Feature Development**:
1. Review AI feature suggestions
2. Add custom processors to `features/feature_processors.py`
3. Update domain patterns in `features/domain_researcher.py`
4. Test with pytest and verify improvement

**Debugging Pipeline Issues**:
1. Check logs in console output
2. Review JSON reports in `outputs/reports/`
3. Verify data files in `outputs/data/`
4. Check feature engineering artifacts in `outputs/features/`
5. Examine error details in failed stage JSON responses

## Environment Variables

**Optional**:
- `OPENAI_API_KEY` - Enable AI advisors (Feature Ideation, Error Analysis, Model Selection)
- `CUDA_VISIBLE_DEVICES` - Control GPU device selection

## Dependencies

**Core**:
- Python 3.10+
- CrewAI 0.193.2+
- pandas, numpy, scikit-learn
- xgboost, catboost, lightgbm

**Optional**:
- OpenAI Python SDK (for AI advisors)
- CUDA + RAPIDS (for GPU acceleration)
- pytest (for testing)

**Installation Options**:
```bash
# Minimal (CPU only)
pip install -r requirements.txt

# Full (GPU + AI)
mamba env create -f environment.yml
conda activate genml
pip install openai
```

## Additional Resources

- **AI Advisors**: See `AI_ADVISORS_README.md` for comprehensive documentation
- **Implementation Details**: See `AI_IMPLEMENTATION_SUMMARY.md`
- **Repository Guidelines**: See `AGENTS.md` for coding standards
- **Setup Instructions**: See `SETUP.md`
- **User Guide**: See `README.md`
