"""
Scikit-learn compatible Random Forest with GPU acceleration
"""
import numpy as np
import warnings
from .backend import get_backend


class RandomForestClassifier:
    """
    GPU-accelerated Random Forest Classifier

    A random forest classifier with optional GPU acceleration using OpenACC.
    Falls back to CPU if GPU is not available.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    max_features : {'sqrt', 'log2', None, int, float}, default='sqrt'
        The number of features to consider when looking for the best split.
        Not yet implemented - uses sqrt by default.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
        Not yet implemented.

    n_jobs : int, default=None
        The number of jobs to run in parallel (CPU mode).
        Not yet implemented.

    random_state : int, default=None
        Controls randomness of the estimator.
        Not yet implemented.

    use_gpu : {'auto', True, False}, default='auto'
        Whether to use GPU acceleration.
        - 'auto': Use GPU if available, otherwise CPU
        - True: Force GPU (raises error if unavailable)
        - False: Force CPU

    verbose : int, default=0
        Controls verbosity of the output.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (Gini importance).

    n_features_in_ : int
        Number of features seen during fit.

    n_classes_ : int
        Number of classes.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    Examples
    --------
    >>> from randomforest_gpu.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=50, use_gpu='auto')
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features='sqrt',
                 bootstrap=True, n_jobs=None, random_state=None,
                 use_gpu='auto', verbose=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.verbose = verbose

        # Attributes set during fit
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.classes_ = None
        self._is_fitted = False

        # Backend
        self._backend = get_backend()

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Not yet implemented.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Store class information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        if sample_weight is not None:
            warnings.warn("sample_weight is not yet implemented and will be ignored")

        if self.verbose:
            backend_type = "GPU" if self._backend.gpu_available else "CPU"
            print(f"Training Random Forest with {self.n_estimators} trees on {backend_type}")

        # Train using backend
        try:
            self.feature_importances_ = self._backend.train(
                X, y,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                use_gpu=self.use_gpu
            )
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        if not self._is_fitted:
            raise RuntimeError("This RandomForestClassifier instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using predict.")

        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but RandomForestClassifier "
                           f"is expecting {self.n_features_in_} features")

        # Get predictions from backend
        predictions = self._backend.predict(X)

        # Map back to original classes
        # (predictions are 1-indexed from Fortran)
        return self.classes_[predictions - 1]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Not yet implemented - returns uniform probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        warnings.warn("predict_proba not yet fully implemented, returning uniform probabilities")

        if not self._is_fitted:
            raise RuntimeError("This RandomForestClassifier instance is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Return uniform probabilities for now
        proba = np.ones((n_samples, self.n_classes_)) / self.n_classes_
        return proba

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Not yet implemented.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)

        if sample_weight is not None:
            warnings.warn("sample_weight is not yet implemented and will be ignored")

        return np.mean(y_pred == y_true)

    @property
    def is_fitted(self):
        """Check if the model is fitted"""
        return self._is_fitted


class RandomForestRegressor:
    """
    GPU-accelerated Random Forest Regressor

    Not yet implemented. Placeholder for future development.
    """

    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, use_gpu='auto'):
        raise NotImplementedError(
            "RandomForestRegressor is not yet implemented. "
            "Please use RandomForestClassifier for classification tasks."
        )
