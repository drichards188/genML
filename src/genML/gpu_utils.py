"""
GPU Utilities for Unified GPU Acceleration Support

This module provides a centralized system for GPU detection, smart library imports,
and GPU memory management. It supports both cuML (RAPIDS) and XGBoost GPU acceleration,
with automatic fallback to CPU-based scikit-learn when GPU is unavailable.

Key Features:
- Unified GPU detection for CUDA, cuML, and XGBoost
- Smart import system: tries cuML first, falls back to scikit-learn
- GPU memory monitoring and management
- Consistent error handling across GPU operations
- Type-compatible interfaces for seamless CPU/GPU switching
"""

import logging
import subprocess
from typing import Dict, Any, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global GPU configuration
_GPU_CONFIG = {
    'cuml_available': False,
    'xgboost_gpu_available': False,
    'cuda_available': False,
    'gpu_memory_gb': 0,
    'gpu_name': None,
    'driver_version': None
}


def detect_cuda() -> Dict[str, Any]:
    """
    Detect CUDA availability and GPU specifications.

    Returns:
        dict: CUDA detection results with GPU specs
    """
    try:
        # Try nvidia-smi to get GPU information
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            output = result.stdout.strip().split(',')
            gpu_name = output[0].strip()
            driver_version = output[1].strip()
            memory_mb = float(output[2].strip().split()[0])
            memory_gb = memory_mb / 1024

            logger.info(f"GPU detected: {gpu_name} (Driver: {driver_version}, Memory: {memory_gb:.1f}GB)")

            return {
                'available': True,
                'gpu_name': gpu_name,
                'driver_version': driver_version,
                'memory_gb': memory_gb
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"nvidia-smi not available: {e}")

    return {
        'available': False,
        'gpu_name': None,
        'driver_version': None,
        'memory_gb': 0
    }


def detect_cuml() -> bool:
    """
    Detect cuML (RAPIDS) availability and test basic functionality.

    Returns:
        bool: True if cuML is available and functional
    """
    try:
        import cuml

        # Simple test: try to import a cuML model to verify GPU accessibility
        # cuML 25.x always runs on GPU by default, so if import succeeds, GPU is accessible
        from cuml.linear_model import LogisticRegression

        logger.info(f"cuML {cuml.__version__} detected and GPU accessible")
        return True
    except ImportError:
        logger.debug("cuML not installed")
        return False
    except Exception as e:
        logger.warning(f"cuML import failed: {e}")
        return False


def detect_xgboost_gpu() -> Tuple[bool, Dict[str, str]]:
    """
    Detect XGBoost GPU support and return configuration.

    Returns:
        tuple: (gpu_available, xgb_params)
    """
    try:
        import xgboost as xgb

        # Test XGBoost GPU support with minimal data
        test_model = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
        test_model.fit([[1], [2]], [0, 1])

        logger.info("XGBoost GPU support detected")
        return True, {'tree_method': 'hist', 'device': 'cuda'}
    except Exception as e:
        logger.debug(f"XGBoost GPU support not available: {e}")
        return False, {}


def initialize_gpu_detection(force_cpu: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive GPU detection and initialize global configuration.

    Args:
        force_cpu: If True, force CPU mode even if GPU is available

    Returns:
        dict: Comprehensive GPU configuration
    """
    global _GPU_CONFIG

    if force_cpu:
        logger.info("GPU detection skipped (force_cpu=True)")
        return _GPU_CONFIG

    # Detect CUDA
    cuda_info = detect_cuda()
    _GPU_CONFIG['cuda_available'] = cuda_info['available']
    _GPU_CONFIG['gpu_name'] = cuda_info['gpu_name']
    _GPU_CONFIG['driver_version'] = cuda_info['driver_version']
    _GPU_CONFIG['gpu_memory_gb'] = cuda_info['memory_gb']

    # Detect cuML only if CUDA is available
    if cuda_info['available']:
        _GPU_CONFIG['cuml_available'] = detect_cuml()
        _GPU_CONFIG['xgboost_gpu_available'], xgb_params = detect_xgboost_gpu()
        _GPU_CONFIG['xgb_params'] = xgb_params
    else:
        _GPU_CONFIG['cuml_available'] = False
        _GPU_CONFIG['xgboost_gpu_available'] = False
        _GPU_CONFIG['xgb_params'] = {}

    # Log summary
    logger.info("="*60)
    logger.info("GPU Detection Summary:")
    logger.info(f"  CUDA Available: {_GPU_CONFIG['cuda_available']}")
    if _GPU_CONFIG['cuda_available']:
        logger.info(f"  GPU: {_GPU_CONFIG['gpu_name']}")
        logger.info(f"  Memory: {_GPU_CONFIG['gpu_memory_gb']:.1f}GB")
        logger.info(f"  cuML Available: {_GPU_CONFIG['cuml_available']}")
        logger.info(f"  XGBoost GPU Available: {_GPU_CONFIG['xgboost_gpu_available']}")
    logger.info("="*60)

    return _GPU_CONFIG


def get_gpu_config() -> Dict[str, Any]:
    """
    Get current GPU configuration.

    Returns:
        dict: GPU configuration dictionary
    """
    return _GPU_CONFIG.copy()


def get_gpu_memory_usage() -> Optional[float]:
    """
    Get current GPU memory usage in GB.

    Returns:
        float or None: Memory usage in GB, or None if unavailable
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            memory_mb = float(result.stdout.strip())
            return memory_mb / 1024
    except Exception:
        pass
    return None


# ============================================================================
# Smart Import System - cuML with scikit-learn fallback
# ============================================================================

def get_linear_model_classifier():
    """Get LogisticRegression (cuML if available, else sklearn)"""
    if _GPU_CONFIG.get('cuml_available', False):
        try:
            from cuml.linear_model import LogisticRegression
            logger.debug("Using cuML LogisticRegression (GPU)")
            return LogisticRegression
        except ImportError:
            pass

    from sklearn.linear_model import LogisticRegression
    logger.debug("Using sklearn LogisticRegression (CPU)")
    return LogisticRegression


def get_linear_model_regressor():
    """Get LinearRegression (cuML if available, else sklearn)"""
    if _GPU_CONFIG.get('cuml_available', False):
        try:
            from cuml.linear_model import LinearRegression
            logger.debug("Using cuML LinearRegression (GPU)")
            return LinearRegression
        except ImportError:
            pass

    from sklearn.linear_model import LinearRegression
    logger.debug("Using sklearn LinearRegression (CPU)")
    return LinearRegression


def get_random_forest_classifier():
    """Get RandomForestClassifier (cuML if available, else sklearn)"""
    if _GPU_CONFIG.get('cuml_available', False):
        try:
            from cuml.ensemble import RandomForestClassifier
            logger.debug("Using cuML RandomForestClassifier (GPU)")
            return RandomForestClassifier
        except ImportError:
            pass

    from sklearn.ensemble import RandomForestClassifier
    logger.debug("Using sklearn RandomForestClassifier (CPU)")
    return RandomForestClassifier


def get_random_forest_regressor():
    """Get RandomForestRegressor (cuML if available, else sklearn)"""
    if _GPU_CONFIG.get('cuml_available', False):
        try:
            from cuml.ensemble import RandomForestRegressor
            logger.debug("Using cuML RandomForestRegressor (GPU)")
            return RandomForestRegressor
        except ImportError:
            pass

    from sklearn.ensemble import RandomForestRegressor
    logger.debug("Using sklearn RandomForestRegressor (CPU)")
    return RandomForestRegressor


def get_standard_scaler():
    """Get StandardScaler (cuML if available, else sklearn)"""
    if _GPU_CONFIG.get('cuml_available', False):
        try:
            from cuml.preprocessing import StandardScaler
            logger.debug("Using cuML StandardScaler (GPU)")
            return StandardScaler
        except ImportError:
            pass

    from sklearn.preprocessing import StandardScaler
    logger.debug("Using sklearn StandardScaler (CPU)")
    return StandardScaler


# ============================================================================
# Data Transfer Utilities (for cuML <-> NumPy conversion)
# ============================================================================

def to_gpu_array(data):
    """
    Convert data to GPU array (CuPy) if cuML is available.

    Args:
        data: NumPy array or pandas DataFrame

    Returns:
        GPU array if cuML available, otherwise original data
    """
    if not _GPU_CONFIG.get('cuml_available', False):
        return data

    try:
        import cupy as cp
        import numpy as np

        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        return data  # DataFrame or other types handled by cuML
    except Exception as e:
        logger.warning(f"GPU array conversion failed: {e}, using CPU")
        return data


def to_cpu_array(data):
    """
    Convert GPU array back to NumPy (CPU) if needed.

    Args:
        data: CuPy array or NumPy array

    Returns:
        NumPy array
    """
    try:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
    except (ImportError, Exception):
        pass
    return data


# ============================================================================
# Convenience Functions
# ============================================================================

def is_gpu_available() -> bool:
    """Check if any GPU acceleration is available"""
    return _GPU_CONFIG.get('cuml_available', False) or _GPU_CONFIG.get('xgboost_gpu_available', False)


def is_cuml_available() -> bool:
    """Check if cuML (RAPIDS) is available"""
    return _GPU_CONFIG.get('cuml_available', False)


def is_xgboost_gpu_available() -> bool:
    """Check if XGBoost GPU is available"""
    return _GPU_CONFIG.get('xgboost_gpu_available', False)


def get_xgboost_params() -> Dict[str, str]:
    """Get XGBoost GPU parameters"""
    return _GPU_CONFIG.get('xgb_params', {})


def log_gpu_memory(stage: str = ""):
    """Log current GPU memory usage"""
    memory_gb = get_gpu_memory_usage()
    if memory_gb is not None:
        logger.info(f"GPU Memory [{stage}]: {memory_gb:.2f}GB / {_GPU_CONFIG['gpu_memory_gb']:.1f}GB")


# Initialize GPU detection on module import
initialize_gpu_detection()
