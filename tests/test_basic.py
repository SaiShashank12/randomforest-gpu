"""
Basic tests for GPU-accelerated Random Forest
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


try:
    from randomforest_gpu import RandomForestClassifier, check_gpu_available
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


@pytest.mark.skipif(not PACKAGE_AVAILABLE, reason="Package not installed")
class TestRandomForestBasic:
    """Basic functionality tests"""

    def test_import(self):
        """Test that package imports correctly"""
        from randomforest_gpu import RandomForestClassifier
        assert RandomForestClassifier is not None

    def test_initialization(self):
        """Test RandomForestClassifier initialization"""
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        assert clf.n_estimators == 10
        assert clf.max_depth == 5
        assert not clf.is_fitted

    def test_fit_basic(self):
        """Test basic fitting"""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X, y)

        assert clf.is_fitted
        assert clf.n_features_in_ == 10
        assert clf.n_classes_ == 2
        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == 10

    def test_predict_shape(self):
        """Test that predictions have correct shape"""
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        assert predictions.shape == (20,)
        assert all(p in clf.classes_ for p in predictions)

    def test_multiclass(self):
        """Test with more than 2 classes"""
        X = np.random.rand(150, 10)
        y = np.random.randint(0, 3, 150)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        predictions = clf.predict(X)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_score(self):
        """Test scoring method"""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=5, use_gpu=False)
        clf.fit(X, y)
        score = clf.score(X, y)

        assert 0 <= score <= 1

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error"""
        clf = RandomForestClassifier()
        X = np.random.rand(10, 5)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)

    def test_wrong_shape_predict_raises(self):
        """Test that wrong shape in predict raises error"""
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 5)  # Wrong number of features

        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X_train, y_train)

        with pytest.raises(ValueError, match="features"):
            clf.predict(X_test)

    def test_feature_importances_sum_to_one(self):
        """Test that feature importances sum to approximately 1"""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=10, use_gpu=False)
        clf.fit(X, y)

        # Allow for small numerical errors
        assert_allclose(np.sum(clf.feature_importances_), 1.0, rtol=0.1)


@pytest.mark.skipif(not PACKAGE_AVAILABLE, reason="Package not installed")
class TestGPUAvailability:
    """Test GPU availability checking"""

    def test_check_gpu_available(self):
        """Test GPU availability check function"""
        gpu_available = check_gpu_available()
        assert isinstance(gpu_available, bool)

    def test_gpu_auto_mode(self):
        """Test that auto mode doesn't crash"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=3, use_gpu='auto')
        clf.fit(X, y)
        assert clf.is_fitted

    def test_cpu_forced_mode(self):
        """Test forcing CPU mode"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=3, use_gpu=False)
        clf.fit(X, y)
        assert clf.is_fitted
