# Building Wheels Locally

This guide explains how to build distribution wheels for randomforest-gpu.

## Quick Build (Current Platform Only)

```bash
# Make script executable
chmod +x build_wheels.sh

# Run build script
./build_wheels.sh
```

This creates:
- `wheelhouse/*.whl` - Binary wheel for your platform
- `wheelhouse/*.tar.gz` - Source distribution

## Manual Build

```bash
# Install build tools
pip install build

# Build wheel and source distribution
python -m build --wheel --outdir wheelhouse
python -m build --sdist --outdir wheelhouse
```

## Build for Multiple Platforms (Using GitHub Actions)

To build wheels for Linux, macOS (Intel + ARM), and Windows:

```bash
# Trigger GitHub Actions workflow manually
gh workflow run wheels.yml

# Or create a release (builds automatically)
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0
```

## Using cibuildwheel (Advanced)

For local multi-platform builds:

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build for Linux (requires Docker)
cibuildwheel --platform linux

# Build for macOS
cibuildwheel --platform macos

# Build for Windows (Windows only)
cibuildwheel --platform windows
```

## Built Wheel Naming

Wheels follow this naming convention:
```
randomforest_gpu-{version}-{python}-{abi}-{platform}.whl
```

Examples:
- `randomforest_gpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl` (macOS ARM Python 3.12)
- `randomforest_gpu-0.1.0-cp312-cp312-manylinux_2_17_x86_64.whl` (Linux Python 3.12)
- `randomforest_gpu-0.1.0-cp312-cp312-win_amd64.whl` (Windows Python 3.12)

## Adding Wheels to Repository

```bash
# Create wheelhouse directory
mkdir -p wheelhouse

# Build wheels
./build_wheels.sh

# Add to git (WARNING: wheels are large binary files)
git add wheelhouse/
git commit -m "Add pre-built wheels for v0.1.0"
git push
```

**Note:** Adding large binary files to git is generally not recommended. Consider:
- Using GitHub Releases to attach wheels
- Publishing to PyPI instead
- Using Git LFS for large files

## Distribution Options

### Option 1: GitHub Releases (Recommended for now)

```bash
# Build wheels
./build_wheels.sh

# Create release and attach wheels
gh release create v0.1.0 wheelhouse/* \
  --title "v0.1.0" \
  --notes "Initial release with pre-built wheels"
```

Users install with:
```bash
# Download and install specific wheel
wget https://github.com/SaiShashank12/randomforest-gpu/releases/download/v0.1.0/randomforest_gpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl
pip install randomforest_gpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl
```

### Option 2: PyPI (Best for users - not set up yet)

Requires PyPI account and credentials. See main README for publishing instructions.

### Option 3: Add to Git Repo (Simple but not ideal)

```bash
# Add .gitattributes for LFS (optional)
echo "*.whl filter=lfs diff=lfs merge=lfs -text" > .gitattributes

# Or just add wheels directly (works for small packages)
git add wheelhouse/
git commit -m "Add wheels"
git push
```

Users install with:
```bash
# Clone and install
git clone https://github.com/SaiShashank12/randomforest-gpu.git
pip install randomforest-gpu/wheelhouse/*.whl
```

## Testing Wheels

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # or `test_env\Scripts\activate` on Windows

# Install wheel
pip install wheelhouse/*.whl

# Test import
python -c "from randomforest_gpu import RandomForestClassifier; print('✓ Success!')"

# Deactivate
deactivate
```

## Current Platform

To see what wheel will be built on your system:

```bash
python -c "import sysconfig; print(f'Platform: {sysconfig.get_platform()}')"
python -c "import sys; print(f'Python: cp{sys.version_info.major}{sys.version_info.minor}')"
```
