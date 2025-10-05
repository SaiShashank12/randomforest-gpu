#!/usr/bin/env python3
"""Test GPU usage with larger dataset"""
import numpy as np
from randomforest_gpu import RandomForestClassifier
import time

# Create a larger dataset to see GPU usage
print("Generating large dataset...")
n_samples = 100000
n_features = 50
n_classes = 10

np.random.seed(42)
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
print("Training Random Forest with GPU acceleration...")
print("Monitor GPU with: nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1")
print()

# Train with many trees to increase GPU workload
rf = RandomForestClassifier(
    n_estimators=100,  # Many trees
    max_depth=10,
    min_samples_split=2,
    use_gpu=True
)

start = time.time()
rf.fit(X, y)
elapsed = time.time() - start

print(f"\nTraining completed in {elapsed:.2f} seconds")
print(f"Feature importances shape: {rf.feature_importances_.shape}")
print(f"First 5 importances: {rf.feature_importances_[:5]}")
