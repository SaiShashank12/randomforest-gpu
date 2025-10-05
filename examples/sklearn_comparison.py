#!/usr/bin/env python
"""
Compare GPU-accelerated Random Forest with scikit-learn
"""
import numpy as np
import time

try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available, skipping comparison")
    exit(0)

from randomforest_gpu import RandomForestClassifier, check_gpu_available

# Generate synthetic dataset
print("Generating dataset...")
X, y = make_classification(
    n_samples=10000,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 3 classes")
print(f"GPU available: {check_gpu_available()}")

# Split data
n_train = 8000
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Parameters for both
n_estimators = 100
max_depth = 15

# Train sklearn version
print("\n" + "="*60)
print("Training scikit-learn RandomForest...")
print("="*60)
start = time.time()
clf_sklearn = SklearnRF(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1  # Use all cores
)
clf_sklearn.fit(X_train, y_train)
sklearn_time = time.time() - start
sklearn_score = clf_sklearn.score(X_test, y_test)

print(f"Training time: {sklearn_time:.2f}s")
print(f"Test accuracy: {sklearn_score:.4f}")

# Train GPU version
print("\n" + "="*60)
print("Training GPU-accelerated RandomForest...")
print("="*60)
start = time.time()
clf_gpu = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    use_gpu='auto',
    verbose=1
)
clf_gpu.fit(X_train, y_train)
gpu_time = time.time() - start
gpu_score = clf_gpu.score(X_test, y_test)

print(f"Training time: {gpu_time:.2f}s")
print(f"Test accuracy: {gpu_score:.4f}")

# Summary
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"sklearn time:  {sklearn_time:.2f}s")
print(f"GPU time:      {gpu_time:.2f}s")
if gpu_time > 0:
    speedup = sklearn_time / gpu_time
    print(f"Speedup:       {speedup:.2f}x")
print()
print(f"sklearn accuracy: {sklearn_score:.4f}")
print(f"GPU accuracy:     {gpu_score:.4f}")
print(f"Difference:       {abs(sklearn_score - gpu_score):.4f}")
