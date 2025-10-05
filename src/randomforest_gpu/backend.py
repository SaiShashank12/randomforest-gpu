"""
Backend module for loading and interfacing with Fortran library
"""
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import sys
import warnings


def find_library():
    """Locate the compiled Fortran library"""
    # Look in package directory
    package_dir = os.path.dirname(__file__)

    # Try different library extensions
    lib_names = [
        'librf_gpu_core.so',      # Linux
        'librf_gpu_core.dylib',   # macOS
        'librf_gpu_core.dll',     # Windows
        'rf_gpu_core.so',
        'rf_gpu_core.dylib',
        'rf_gpu_core.dll',
    ]

    for lib_name in lib_names:
        lib_path = os.path.join(package_dir, lib_name)
        if os.path.exists(lib_path):
            return lib_path

    # Also check in build directory for development
    build_paths = [
        os.path.join(package_dir, '../../build'),
        os.path.join(package_dir, '../../builddir'),
    ]

    for build_path in build_paths:
        if os.path.exists(build_path):
            for lib_name in lib_names:
                lib_path = os.path.join(build_path, lib_name)
                if os.path.exists(lib_path):
                    return lib_path

    return None


class FortranBackend:
    """Interface to Fortran GPU-accelerated random forest library"""

    def __init__(self):
        self.lib = None
        self.gpu_available = False
        self._load_library()

    def _load_library(self):
        """Load the Fortran shared library"""
        lib_path = find_library()

        if lib_path is None:
            warnings.warn(
                "Could not find compiled Fortran library. "
                "GPU acceleration will not be available. "
                "Please ensure the package is properly installed."
            )
            return

        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
            self._check_gpu_availability()
        except Exception as e:
            warnings.warn(f"Failed to load Fortran library: {e}")

    def _setup_function_signatures(self):
        """Define ctypes signatures for Fortran functions"""
        if self.lib is None:
            return

        # train_rf_gpu signature
        self.lib.train_rf_gpu.argtypes = [
            ndpointer(dtype=np.float64, ndim=2, flags='F'),  # x (Fortran order)
            ndpointer(dtype=np.float64, ndim=1, flags='C'),  # y
            ndpointer(dtype=np.int32, ndim=1, flags='C'),    # classes
            ctypes.c_int,  # n
            ctypes.c_int,  # p
            ctypes.c_int,  # nclass
            ctypes.c_int,  # n_estimators
            ctypes.c_int,  # max_depth
            ctypes.c_int,  # min_samples_split
            ndpointer(dtype=np.float64, ndim=1, flags='C'),  # feature_importances
            ctypes.c_int,  # use_gpu
        ]
        self.lib.train_rf_gpu.restype = None

        # predict_rf_gpu signature
        self.lib.predict_rf_gpu.argtypes = [
            ndpointer(dtype=np.float64, ndim=2, flags='F'),  # x_test
            ctypes.c_int,  # n_test
            ctypes.c_int,  # p
            ndpointer(dtype=np.int32, ndim=1, flags='C'),    # predictions
        ]
        self.lib.predict_rf_gpu.restype = None

        # check_gpu_available signature
        self.lib.check_gpu_available.argtypes = []
        self.lib.check_gpu_available.restype = ctypes.c_int

    def _check_gpu_availability(self):
        """Check if GPU is available"""
        if self.lib is None:
            self.gpu_available = False
            return

        try:
            result = self.lib.check_gpu_available()
            self.gpu_available = (result == 1)
        except Exception as e:
            warnings.warn(f"GPU check failed: {e}")
            self.gpu_available = False

    def train(self, X, y, n_estimators=100, max_depth=None,
              min_samples_split=2, use_gpu='auto'):
        """
        Train random forest

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        n_estimators : int, default=100
            Number of trees
        max_depth : int, default=None
            Maximum tree depth (None = unlimited, set to 1000)
        min_samples_split : int, default=2
            Minimum samples to split a node
        use_gpu : {'auto', True, False}, default='auto'
            Whether to use GPU acceleration

        Returns
        -------
        feature_importances : ndarray of shape (n_features,)
            Feature importance scores
        """
        if self.lib is None:
            raise RuntimeError("Fortran library not loaded")

        # Convert inputs
        X = np.asarray(X, dtype=np.float64, order='F')  # Fortran order
        y = np.asarray(y, dtype=np.float64)

        n, p = X.shape

        # Convert classes to integers (0-indexed for Fortran)
        classes_unique = np.unique(y)
        nclass = len(classes_unique)
        class_map = {c: i+1 for i, c in enumerate(classes_unique)}  # 1-indexed
        classes = np.array([class_map[val] for val in y], dtype=np.int32)

        # Handle max_depth
        if max_depth is None:
            max_depth = 1000

        # Determine GPU usage
        if use_gpu == 'auto':
            use_gpu_flag = 1 if self.gpu_available else 0
        elif use_gpu:
            if not self.gpu_available:
                warnings.warn("GPU requested but not available, using CPU")
            use_gpu_flag = 1 if self.gpu_available else 0
        else:
            use_gpu_flag = 0

        # Allocate output
        feature_importances = np.zeros(p, dtype=np.float64)

        # Call Fortran
        self.lib.train_rf_gpu(
            X, y, classes,
            n, p, nclass,
            n_estimators, max_depth, min_samples_split,
            feature_importances,
            use_gpu_flag
        )

        return feature_importances

    def predict(self, X):
        """
        Make predictions

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if self.lib is None:
            raise RuntimeError("Fortran library not loaded")

        X = np.asarray(X, dtype=np.float64, order='F')
        n_test, p = X.shape

        predictions = np.zeros(n_test, dtype=np.int32)

        self.lib.predict_rf_gpu(X, n_test, p, predictions)

        return predictions


# Global backend instance
_backend = None


def get_backend():
    """Get or create the global backend instance"""
    global _backend
    if _backend is None:
        _backend = FortranBackend()
    return _backend
