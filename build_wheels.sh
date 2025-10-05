#!/bin/bash
# Build wheels locally for distribution

set -e

echo "Building randomforest-gpu wheels..."
echo "===================================="

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from randomforest-gpu directory"
    exit 1
fi

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip build

# Create wheelhouse directory
mkdir -p wheelhouse

# Build wheel and source distribution
echo ""
echo "Building wheel for current platform..."
python -m build --wheel --outdir wheelhouse

echo ""
echo "Building source distribution..."
python -m build --sdist --outdir wheelhouse

echo ""
echo "✓ Build complete!"
echo ""
echo "Wheels created in: wheelhouse/"
ls -lh wheelhouse/

echo ""
echo "To install locally:"
echo "  pip install wheelhouse/*.whl"
echo ""
echo "To add to git repo:"
echo "  git add wheelhouse/"
echo "  git commit -m 'Add pre-built wheels'"
echo "  git push"
