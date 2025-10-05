#!/bin/bash
# Script to install gfortran on macOS for wheel building
set -xe

PLAT=$1
PROJECT_DIR=$2

# Install gfortran for specific architecture
if [[ $PLAT == "arm64" ]]; then
    # ARM64 (Apple Silicon)
    echo "Installing gfortran for ARM64..."
    brew install gcc@13
    export FC=gfortran-13
else
    # x86_64 (Intel)
    echo "Installing gfortran for x86_64..."
    brew install gcc@13
    export FC=gfortran-13
fi

export LDFLAGS="-L/opt/homebrew/lib -L/usr/local/lib"
export CPPFLAGS="-I/opt/homebrew/include -I/usr/local/include"

# Verify installation
which gfortran-13
gfortran-13 --version
