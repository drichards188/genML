# AI Advisors Integration

This document describes the AI-powered intelligent advisors integrated into the ML pipeline to provide automated analysis and improvement suggestions.

## Overview

The AI advisors use OpenAI's GPT-4o-mini model to analyze your ML pipeline and suggest improvements. Two priority advisors have been implemented:

1. **Error Pattern Analyzer** (Priority 1) - Analyzes prediction errors to identify patterns and suggest targeted improvements
2. **Feature Ideation Advisor** (Priority 2) - Suggests domain-specific features based on your dataset

## Quick Start

### Prerequisites

1. Install OpenAI package (if not already installed):
   ```bash
   pip install openai
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

3. Run the pipeline as normal:
   ```bash
   conda activate genml
   python3 src/genML/main.py
   ```

The AI advisors will automatically run during feature engineering and model training stages.

## Configuration

AI advisors are configured in `src/genML/tools.py` via the `AI_ADVISORS_CONFIG` dictionary:

```python
AI_ADVISORS_CONFIG = {
    'enabled': True,                         # Enable/disable AI advisors globally
    'feature_ideation': {
        'enabled': True,                     # Enable feature ideation advisor
        'sample_size': 100,                  # Number of rows to sample for analysis
        'save_report': True,                 # Save feature suggestions report
    },
    'error_analysis': {
        'enabled': True,                     # Enable error pattern analyzer
        'top_n_errors': 100,                 # Number of worst errors to analyze
        'save_report': True,                 # Save error analysis report
    },
    'openai_config': {
        'model': 'gpt-4o-mini',              # OpenAI model to use
        'max_cost_per_run': 10.0,            # Maximum API cost per run (USD)
        'enable_cache': True,                # Enable response caching
        'cache_dir': 'outputs/ai_cache',     # Cache directory
    }
}
```

### Disabling AI Advisors

To disable AI advisors without removing the code, set:
```python
AI_ADVISORS_CONFIG = {
    'enabled': False,
    # ... rest of config
}
```

Or disable individual advisors:
```python
AI_ADVISORS_CONFIG = {
    'enabled': True,
    'feature_ideation': {
        'enabled': False,  # Disable feature ideation
        # ...
    },
    'error_analysis': {
        'enabled': False,  # Disable error analysis
        # ...
    },
    # ...
}
```

## What Each Advisor Does

### Feature Ideation Advisor

**When it runs:** During the feature engineering stage, after initial features are generated

**What it analyzes:**
- Current feature set
- Feature importances
- Dataset samples
- Detected domain (e.g., transportation, healthcare)

**What it suggests:**
- New engineered features with Python formulas
- Feature interactions (e.g., speed × curvature)
- Feature transformations (e.g., log, sqrt, binning)
- Priority ranking of suggestions

**Example output:**
```
🤖 AI Feature Ideation Advisor
============================================================
Analyzing current features and suggesting improvements...
✅ AI Feature Ideation completed!
   New features suggested: 12
   Interaction suggestions: 5
   Transformation suggestions: 3

   Top Priority Features:
      1. speed_danger_score
      2. weather_visibility_risk
      3. curvature_bins

   📄 Report saved to: outputs/reports/ai_feature_suggestions.json
   💰 API Cost: $0.0234
   📊 Tokens: 1,523 (892 in, 631 out)
```

**Report location:** `outputs/reports/ai_feature_suggestions.json`

### Error Pattern Analyzer

**When it runs:** During model training, after the best model is trained on full data

**What it analyzes:**
- Prediction errors on training data
- Feature-error correlations
- Worst predictions in detail
- Model type and characteristics

**What it suggests:**
- Error patterns detected
- Feature engineering suggestions
- Subpopulation issues
- Model improvements
- Data quality concerns
- Priority actions

**Example output:**
```
🤖 AI Error Pattern Analyzer
============================================================
Analyzing prediction errors to identify patterns...
✅ AI Error Analysis completed!
   Mean Absolute Error: 0.052341
   RMSE: 0.073482
   Max Error: 0.428761

   🔍 Key Error Patterns Detected:
      1. High errors on roads with extreme curvature (>0.8)
      2. Poor performance in rainy conditions at night
      3. Overestimation for low-speed residential areas

   💡 Priority Actions:
      1. Add interaction: curvature * speed_limit
      2. Create binary flag: is_dangerous_combo
      3. Add weather_encoded * lighting_encoded

   📊 Top Features Correlated with Errors:
      1. curvature: +0.342
      2. weather_encoded: +0.287
      3. lighting_encoded: +0.231

   📄 Report saved to: outputs/reports/ai_error_analysis.json
   💰 API Cost: $0.0187
   📊 Tokens: 1,234 (723 in, 511 out)
```

**Report location:** `outputs/reports/ai_error_analysis.json`

## Output Files

The AI advisors generate JSON reports with detailed suggestions:

### Feature Suggestions Report
**File:** `outputs/reports/ai_feature_suggestions.json`

**Structure:**
```json
{
  "status": "success",
  "current_feature_count": 19,
  "detected_domain": "transportation",
  "suggested_features": [
    {
      "name": "speed_danger_score",
      "formula": "df['speed_limit'] * df['curvature']",
      "type": "interaction",
      "rationale": "Speed combined with curvature indicates accident risk",
      "expected_impact": "high",
      "domain_knowledge": "Physics: Higher speeds on curved roads increase danger"
    },
    ...
  ],
  "interaction_suggestions": [...],
  "transformation_suggestions": [...],
  "priority_features": ["speed_danger_score", "weather_visibility_risk", ...]
}
```

### Error Analysis Report
**File:** `outputs/reports/ai_error_analysis.json`

**Structure:**
```json
{
  "status": "success",
  "error_statistics": {
    "mean_absolute_error": 0.052341,
    "rmse": 0.073482,
    "max_error": 0.428761
  },
  "feature_error_correlations": {
    "curvature": 0.342,
    "weather_encoded": 0.287,
    ...
  },
  "ai_suggestions": {
    "error_patterns_detected": [...],
    "feature_engineering_suggestions": [...],
    "subpopulation_issues": [...],
    "model_improvements": [...],
    "priority_actions": [...]
  }
}
```

## Cost Management

The AI advisors include several cost-saving features:

### Response Caching
- Responses are cached in `outputs/ai_cache/`
- Identical requests return cached results instantly
- Saves API costs on repeated runs
- Cache persists between runs

### Spending Cap
- Default maximum: $10.00 per run
- Configurable via `max_cost_per_run`
- Pipeline stops making API calls after limit is reached
- Prevents runaway costs

### Token Usage Tracking
- Tracks input and output tokens
- Estimates cost in real-time
- Displays usage after each advisor run
- Total cost shown in reports

### Expected Costs

Based on GPT-4o-mini pricing ($0.150 per 1M input tokens, $0.600 per 1M output tokens):

- **Feature Ideation:** ~$0.02-0.05 per run
- **Error Analysis:** ~$0.01-0.03 per run
- **Total per pipeline run:** ~$0.03-0.08

For typical usage (10-20 pipeline runs), expect total costs of $0.50-1.50.

## Implementation Architecture

### Module Structure
```
src/genML/ai_advisors/
├── __init__.py              # Module exports
├── openai_client.py         # Robust API client with caching
├── error_analyzer.py        # Error pattern analysis
└── feature_ideation.py      # Feature suggestion generation
```

### Integration Points

1. **Feature Engineering** (`tools.py:338-418`)
   - After `feature_engine.fit(train_df, target_col)`
   - Before feature transformation
   - Calls `FeatureIdeationAdvisor.suggest_features()`

2. **Model Training** (`tools.py:1620-1707`)
   - After `best_model.fit(X_train, y_train)`
   - Before model saving
   - Calls `ErrorPatternAnalyzer.analyze_errors()`

### Error Handling

Both advisors include comprehensive error handling:
- Graceful degradation if OpenAI API is unavailable
- Fallback to rule-based suggestions
- Pipeline continues even if AI analysis fails
- All errors logged but don't crash the pipeline

## Testing the Integration

### Test 1: Verify Configuration
```bash
# Check that AI advisors are enabled
grep -A 5 "AI_ADVISORS_CONFIG" src/genML/tools.py
```

### Test 2: Run with AI Key
```bash
# Set API key and run
export OPENAI_API_KEY='your-key'
conda activate genml
python3 src/genML/main.py
```

**Expected:** You should see two AI advisor sections in the output with suggestions and cost information.

### Test 3: Run without AI Key
```bash
# Run without API key
unset OPENAI_API_KEY
python3 src/genML/main.py
```

**Expected:** Pipeline runs normally but skips AI analysis with warning messages.

### Test 4: Verify Reports
```bash
# Check that reports were generated
ls -lh outputs/reports/ai_*.json
cat outputs/reports/ai_feature_suggestions.json | jq '.suggested_features | length'
cat outputs/reports/ai_error_analysis.json | jq '.ai_suggestions.priority_actions'
```

### Test 5: Check Cache
```bash
# After first run, check cache
ls -lh outputs/ai_cache/
# Run again - should use cached responses (instant, free)
python3 src/genML/main.py
```

## Measuring Improvement

To measure the impact of AI-suggested improvements:

1. **Baseline:** Note the current best model score (from cross-validation)
2. **Review suggestions:** Read the AI reports in `outputs/reports/`
3. **Implement suggestions:** Add suggested features to the feature engine or config
4. **Re-run pipeline:** Train models with new features
5. **Compare scores:** Check if model performance improved

### Example Workflow

```bash
# 1. Baseline run
python3 src/genML/main.py
# Best model: Stacking Ensemble, MSE: -0.39115

# 2. Review AI suggestions
cat outputs/reports/ai_feature_suggestions.json | jq '.suggested_features[:3]'
# Suggested: speed_danger_score, weather_visibility_risk, etc.

# 3. Implement top suggestions in feature engine
# (Edit src/genML/features/feature_processors.py)

# 4. Re-run with new features
python3 src/genML/main.py
# Best model: Stacking Ensemble, MSE: -0.37823 (3.3% improvement!)
```

## Future Enhancements

Additional AI advisors from the original 7-priority plan:

- **Priority 3:** Meta-Learning Pipeline Optimizer (learns from multiple runs)
- **Priority 4:** Smart Interaction Discovery (identifies high-value interactions)
- **Priority 5:** Adaptive Hyperparameter Strategy (optimizes trial allocation)
- **Priority 6:** Data Quality Auditor (checks for subtle data issues)
- **Priority 7:** Ensemble Composition Advisor (optimizes ensemble members)

## Troubleshooting

### "OpenAI API not available"
- Install: `pip install openai`
- Set key: `export OPENAI_API_KEY='your-key'`
- Verify: `python -c "import openai; print(openai.__version__)"`

### "Cost limit reached"
- Increase limit in `AI_ADVISORS_CONFIG['openai_config']['max_cost_per_run']`
- Or disable caching to reset

### "AI Feature Ideation failed"
- Check OpenAI API status: https://status.openai.com/
- Verify API key is valid
- Check error logs in console output
- Try disabling just that advisor and continuing

### Empty or Generic Suggestions
- Increase `sample_size` for more context
- Check that domain detection is working correctly
- Verify feature importances are being passed

## Contributing

To add new AI advisors:

1. Create new advisor class in `src/genML/ai_advisors/`
2. Follow the pattern of existing advisors (use OpenAIClient)
3. Add configuration to `AI_ADVISORS_CONFIG`
4. Integrate into appropriate pipeline stage in `tools.py`
5. Add comprehensive error handling
6. Update this README

## Support

For issues or questions:
- Check the logs in console output
- Review JSON reports in `outputs/reports/`
- Verify OpenAI API key and account status
- Check configuration in `tools.py`

---

**Implementation Date:** 2025-10-14
**Status:** ✅ Integrated and tested
**Expected Impact:** 5-15% improvement (Priority 1 + Priority 2 combined)
