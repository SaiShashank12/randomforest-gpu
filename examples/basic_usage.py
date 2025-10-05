#!/usr/bin/env python
"""
Basic usage example for GPU-accelerated Random Forest
"""
import numpy as np
from randomforest_gpu import RandomForestClassifier, check_gpu_available

# Check if GPU is available
print(f"GPU available: {check_gpu_available()}")

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
# Create somewhat separable classes
y = (X[:, 0] + X[:, 1] > 0).astype(int)

print(f"Dataset: {n_samples} samples, {n_features} features, 2 classes")

# Split into train and test
n_train = 800
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Create and train classifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    use_gpu='auto',  # Automatically use GPU if available
    verbose=1
)

print("\nTraining Random Forest...")
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print(f"\nResults:")
print(f"  Training accuracy: {train_score:.3f}")
print(f"  Test accuracy: {test_score:.3f}")

# Feature importances
print(f"\nTop 5 most important features:")
importances = clf.feature_importances_
top_features = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(top_features, 1):
    print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")
