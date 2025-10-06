"""
Test that the built wheel loads correctly and Fortran library is accessible.
"""
import pytest
import numpy as np


class TestWheelLoading:
    """Tests for wheel loading and library availability"""

    def test_package_imports(self):
        """Test that package can be imported without errors"""
        try:
            import randomforest_gpu
            from randomforest_gpu import RandomForestClassifier
            assert RandomForestClassifier is not None
        except Exception as e:
            pytest.fail(f"Failed to import package: {e}")

    def test_fortran_library_loads(self):
        """Test that the Fortran backend library loads successfully"""
        from randomforest_gpu.backend import FortranBackend

        backend = FortranBackend()
        assert backend.lib is not None, "Fortran library failed to load"

    def test_library_functions_exist(self):
        """Test that required Fortran functions are available"""
        from randomforest_gpu.backend import FortranBackend

        backend = FortranBackend()

        # Check that critical functions exist
        assert hasattr(backend.lib, 'train_rf_gpu'), "train_rf_gpu function not found"
        assert hasattr(backend.lib, 'predict_rf_gpu'), "predict_rf_gpu function not found"
        assert hasattr(backend.lib, 'check_gpu_available'), "check_gpu_available function not found"

    def test_basic_training_works(self):
        """Test that basic model training works with loaded library"""
        from randomforest_gpu import RandomForestClassifier

        # Simple dataset
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([0, 1, 0, 1], dtype=np.int32)

        # Create and train model
        rf = RandomForestClassifier(n_estimators=2, max_depth=2)

        try:
            rf.fit(X, y)
            assert rf._is_fitted, "Model should be fitted"
            assert rf.feature_importances_ is not None, "Feature importances should be set"
        except Exception as e:
            pytest.fail(f"Training failed with loaded library: {e}")

    def test_prediction_works(self):
        """Test that prediction works with loaded library"""
        from randomforest_gpu import RandomForestClassifier

        # Simple dataset
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([0, 1, 0, 1], dtype=np.int32)

        # Train model
        rf = RandomForestClassifier(n_estimators=2, max_depth=2)
        rf.fit(X, y)

        # Predict
        try:
            predictions = rf.predict(X)
            assert len(predictions) == len(y), "Predictions should match input size"
            assert all(p in [0, 1] for p in predictions), "Predictions should be valid classes"
        except Exception as e:
            pytest.fail(f"Prediction failed with loaded library: {e}")

    def test_no_library_warnings(self):
        """Test that no library loading warnings are raised"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from randomforest_gpu import RandomForestClassifier

            # Check if any warnings about library loading
            library_warnings = [warning for warning in w
                              if "Failed to load Fortran library" in str(warning.message)]

            assert len(library_warnings) == 0, \
                f"Library loading warnings detected: {[str(w.message) for w in library_warnings]}"

    def test_ctypes_argtypes_configured(self):
        """Test that ctypes argument types are properly configured"""
        from randomforest_gpu.backend import FortranBackend

        backend = FortranBackend()

        # Check that argtypes are set (means library loaded successfully)
        assert backend.lib is not None, "Library not loaded"
        assert backend.lib.train_rf_gpu.argtypes is not None, \
            "train_rf_gpu argtypes not configured"
        assert backend.lib.predict_rf_gpu.argtypes is not None, \
            "predict_rf_gpu argtypes not configured"
