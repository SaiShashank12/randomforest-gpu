"""
Tests comparing GPU and CPU implementations for numerical accuracy
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose


try:
    from randomforest_gpu import RandomForestClassifier, check_gpu_available
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


GPU_AVAILABLE = PACKAGE_AVAILABLE and check_gpu_available() if PACKAGE_AVAILABLE else False


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUCPUParity:
    """Compare GPU and CPU implementations"""

    def test_feature_importances_similar(self):
        """Test that GPU and CPU produce similar feature importances"""
        np.random.seed(42)
        X = np.random.rand(200, 20)
        y = np.random.randint(0, 2, 200)

        # CPU version
        clf_cpu = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            random_state=42,
            use_gpu=False
        )
        clf_cpu.fit(X, y)

        # GPU version
        clf_gpu = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            random_state=42,
            use_gpu=True
        )
        clf_gpu.fit(X, y)

        # Feature importances should be similar
        # Allow for reasonable tolerance due to floating-point differences
        assert_allclose(
            clf_cpu.feature_importances_,
            clf_gpu.feature_importances_,
            rtol=0.1,  # 10% relative tolerance
            atol=0.01  # 1% absolute tolerance
        )

    def test_predictions_similar(self):
        """Test that predictions are similar between GPU and CPU"""
        np.random.seed(42)
        X_train = np.random.rand(200, 20)
        y_train = np.random.randint(0, 2, 200)
        X_test = np.random.rand(50, 20)

        # CPU version
        clf_cpu = RandomForestClassifier(n_estimators=10, use_gpu=False)
        clf_cpu.fit(X_train, y_train)
        pred_cpu = clf_cpu.predict(X_test)

        # GPU version
        clf_gpu = RandomForestClassifier(n_estimators=10, use_gpu=True)
        clf_gpu.fit(X_train, y_train)
        pred_gpu = clf_gpu.predict(X_test)

        # At least 80% of predictions should match
        agreement = np.mean(pred_cpu == pred_gpu)
        assert agreement >= 0.8, f"Only {agreement:.1%} predictions match"

    def test_score_similar(self):
        """Test that scores are similar between GPU and CPU"""
        np.random.seed(42)
        X = np.random.rand(200, 20)
        y = np.random.randint(0, 2, 200)

        clf_cpu = RandomForestClassifier(n_estimators=10, use_gpu=False)
        clf_cpu.fit(X, y)
        score_cpu = clf_cpu.score(X, y)

        clf_gpu = RandomForestClassifier(n_estimators=10, use_gpu=True)
        clf_gpu.fit(X, y)
        score_gpu = clf_gpu.score(X, y)

        # Scores should be within 10%
        assert abs(score_cpu - score_gpu) < 0.1


@pytest.mark.skipif(not PACKAGE_AVAILABLE, reason="Package not installed")
class TestNumericalStability:
    """Test numerical stability and edge cases"""

    def test_handle_nan_gracefully(self):
        """Test that NaN values are handled appropriately"""
        X = np.random.rand(100, 10)
        X[0, 0] = np.nan
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)

        # Should either handle NaN or raise clear error
        try:
            clf.fit(X, y)
            predictions = clf.predict(X)
            assert len(predictions) == 100
        except (ValueError, RuntimeError):
            pass  # Acceptable to reject NaN values

    def test_small_dataset(self):
        """Test with very small dataset"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        clf = RandomForestClassifier(n_estimators=2, use_gpu=False)
        clf.fit(X, y)
        predictions = clf.predict(X)

        assert len(predictions) == 3

    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.rand(100, 1)
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X, y)
        assert clf.n_features_in_ == 1

    def test_large_number_of_classes(self):
        """Test with many classes"""
        X = np.random.rand(500, 10)
        y = np.random.randint(0, 10, 500)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X, y)
        assert clf.n_classes_ == 10
