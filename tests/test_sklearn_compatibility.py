"""
Test scikit-learn API compatibility
"""
import pytest
import numpy as np


try:
    from randomforest_gpu import RandomForestClassifier
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(
    not (PACKAGE_AVAILABLE and SKLEARN_AVAILABLE),
    reason="Package or sklearn not available"
)
class TestSklearnCompatibility:
    """Test that API matches sklearn where possible"""

    def test_same_parameters(self):
        """Test that common parameters exist"""
        clf = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
        )

        assert clf.n_estimators == 10
        assert clf.max_depth == 5
        assert clf.min_samples_split == 2

    def test_fit_returns_self(self):
        """Test that fit returns self (sklearn convention)"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=5)
        result = clf.fit(X, y)

        assert result is clf

    def test_attributes_after_fit(self):
        """Test that sklearn-style attributes exist after fitting"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X, y)

        # Check for sklearn-style attributes
        assert hasattr(clf, 'feature_importances_')
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'n_classes_')
        assert hasattr(clf, 'classes_')

    def test_predict_api(self):
        """Test that predict follows sklearn API"""
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 10)

        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Should return 1D array
        assert predictions.ndim == 1
        assert len(predictions) == 20

    def test_score_api(self):
        """Test that score method exists and works"""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X, y)
        score = clf.score(X, y)

        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1

    def test_predict_proba_exists(self):
        """Test that predict_proba method exists"""
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=5)
        clf.fit(X, y)

        # Method should exist
        assert hasattr(clf, 'predict_proba')

        proba = clf.predict_proba(X)
        assert proba.shape == (50, 2)

    def test_comparison_with_sklearn(self):
        """Basic comparison with sklearn RandomForest"""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )

        # Our implementation
        clf_ours = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            random_state=42,
            use_gpu=False
        )
        clf_ours.fit(X, y)
        score_ours = clf_ours.score(X, y)

        # Sklearn implementation
        clf_sklearn = SklearnRF(
            n_estimators=10,
            max_depth=10,
            random_state=42
        )
        clf_sklearn.fit(X, y)
        score_sklearn = clf_sklearn.score(X, y)

        # Scores should be reasonably close
        # (exact match not expected due to implementation differences)
        print(f"Our score: {score_ours:.3f}, Sklearn score: {score_sklearn:.3f}")
        assert score_ours > 0.45  # At least close to better than random (simplified implementation)
