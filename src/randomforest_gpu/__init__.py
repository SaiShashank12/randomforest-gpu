"""
GPU-Accelerated Random Forest
==============================

A GPU-accelerated random forest implementation using Fortran with OpenACC
directives, providing a scikit-learn compatible API.

Features
--------
- GPU acceleration using OpenACC (NVIDIA GPUs)
- Automatic CPU fallback when GPU unavailable
- Scikit-learn compatible API
- Based on original Breiman-Cutler Fortran implementation

Examples
--------
>>> from randomforest_gpu import RandomForestClassifier
>>> import numpy as np
>>> X = np.random.rand(1000, 20)
>>> y = np.random.randint(0, 2, 1000)
>>> clf = RandomForestClassifier(n_estimators=100, use_gpu='auto')
>>> clf.fit(X, y)
>>> predictions = clf.predict(X)
>>> print(f"Accuracy: {clf.score(X, y):.2f}")
"""

__version__ = "0.1.0"

from .ensemble import RandomForestClassifier, RandomForestRegressor
from .backend import get_backend

__all__ = [
    'RandomForestClassifier',
    'RandomForestRegressor',
    'get_backend',
]


def check_gpu_available():
    """
    Check if GPU acceleration is available

    Returns
    -------
    available : bool
        True if GPU is available, False otherwise
    """
    backend = get_backend()
    return backend.gpu_available


def get_version():
    """Get package version"""
    return __version__
