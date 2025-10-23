# Setup Instructions for genML

This guide will help you set up the genML environment with RAPIDS cuML for GPU-accelerated machine learning.

## Prerequisites

- **NVIDIA GPU** with CUDA support (Compute Capability 7.0+)
- **CUDA Toolkit** 11.8 or 12.0+ installed
- **Anaconda** or **Miniconda** installed

### Check Your GPU

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check CUDA version
nvcc --version
```

## Installation Steps

### 1. Install Mamba (Faster than conda)

```bash
# Install mamba in base environment
conda install -c conda-forge mamba
```

### 2. Create the Environment

```bash
# Navigate to project directory
cd /path/to/genML

# Create environment from environment.yml
mamba env create -f environment.yml
```

This will:
- Install Python 3.10-3.12
- Install RAPIDS cuML with CUDA dependencies
- Install all ML libraries (pandas, scikit-learn, xgboost, etc.)
- Install CrewAI and Optuna via pip

### 3. Activate the Environment

```bash
conda activate genml
```

### 4. Verify Installation

```bash
# Check cuML installation
python -c "import cuml; print(f'cuML version: {cuml.__version__}')"

# Check CUDA is accessible
python -c "import cuml; from cuml.common.device_selection import using_device_type; print(f'GPU available: {using_device_type(\"gpu\")}')"

# Check other packages
python -c "import crewai, pandas, xgboost, optuna; print('All packages loaded successfully')"
```

## Common Issues

### Issue: CUDA version mismatch
**Solution**: Update the `cuda-version` in `environment.yml` to match your system:
```yaml
# For CUDA 12.x
- cuda-version>=12.0,<13.0

# For CUDA 11.x
- cuda-version>=11.8,<12.0
```

### Issue: Out of memory errors
**Solution**: Reduce batch sizes or use CPU fallback:
```python
# In your code, add fallback
try:
    from cuml import RandomForestClassifier  # GPU
except ImportError:
    from sklearn.ensemble import RandomForestClassifier  # CPU
```

### Issue: Slow environment creation
**Solution**: This is normal. RAPIDS packages are large (~2GB). First-time setup takes 10-15 minutes.

## Updating the Environment

When dependencies change:

```bash
# Update environment
mamba env update -f environment.yml --prune

# Or recreate from scratch
mamba env remove -n genml
mamba env create -f environment.yml
```

## Running the Pipeline

```bash
# Activate environment
conda activate genml

# Run the pipeline
python src/genML/main.py
```

## Alternative: CPU-Only Installation

If you don't have a GPU or want CPU-only mode:

```bash
# Use pip instead
pip install -r requirements.txt
```

Note: This will not include RAPIDS cuML and will be significantly slower on large datasets.

## Environment Export

To share your exact environment:

```bash
# Export with versions locked
mamba env export > environment_locked.yml

# Or export without versions
mamba env export --from-history > environment_minimal.yml
```

## Troubleshooting

### List environments
```bash
conda env list
```

### Remove environment
```bash
mamba env remove -n genml
```

### Clear conda cache
```bash
conda clean --all
```

### Check channel priorities
```bash
conda config --show channels
```

## Additional Resources

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [RAPIDS Installation Guide](https://rapids.ai/start.html)
