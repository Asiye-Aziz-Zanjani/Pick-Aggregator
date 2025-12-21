# Installation Guide for Picker Aggregator

This guide provides detailed installation instructions for different operating systems and computing environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Linux/Mac Installation](#linuxmac-installation)
3. [Windows Installation](#windows-installation)
4. [HPC/Cluster Installation](#hpccluster-installation)
5. [GPU Setup](#gpu-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 50 GB free space for data caching
- **Internet**: For downloading seismic data and models

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (significant speedup)
- **RAM**: 32 GB or more
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: SSD for faster data I/O

## Linux/Mac Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv
sudo apt-get install build-essential gfortran libopenblas-dev
```

**macOS (using Homebrew):**
```bash
brew install python@3.10
brew install openblas
```

### Step 2: Create Virtual Environment

```bash
# Create a dedicated directory for the project
mkdir -p ~/seismic_analysis
cd ~/seismic_analysis

# Create virtual environment
python3 -m venv ~/.venv/seisbench

# Activate the environment
source ~/.venv/seisbench/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Python Packages

```bash
# Install core scientific packages
pip install numpy scipy pandas matplotlib seaborn

# Install seismology-specific packages
pip install obspy pyproj tqdm

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR for GPU version (if you have CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SeisBench
pip install git+https://github.com/seisbench/seisbench

# Install GAMMA
pip install git+https://github.com/AI4EPS/GaMMA.git
```

### Step 4: Download the Code

```bash
# Clone or download the repository
git clone https://github.com/yourusername/picker-aggregator.git
cd picker-aggregator

# OR if you have the file directly
# Place picker-aggregator.py in your working directory
```

### Step 5: Configure SeisBench Cache

```bash
# Create SeisBench cache directory
mkdir -p ~/.seisbench

# Set environment variable (add to ~/.bashrc or ~/.bash_profile for persistence)
export SEISBENCH_CACHE_ROOT=~/.seisbench
```

## Windows Installation

### Step 1: Install Python

1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Complete the installation

### Step 2: Install Git (Optional but Recommended)

Download and install from [git-scm.com](https://git-scm.com/download/win)

### Step 3: Create Virtual Environment

Open Command Prompt or PowerShell:

```cmd
# Create project directory
mkdir C:\seismic_analysis
cd C:\seismic_analysis

# Create virtual environment
python -m venv seisbench_env

# Activate the environment
seisbench_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Packages

```cmd
# Install core packages
pip install numpy scipy pandas matplotlib seaborn obspy pyproj tqdm

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# OR for GPU version (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SeisBench
pip install git+https://github.com/seisbench/seisbench

# Install GAMMA
pip install git+https://github.com/AI4EPS/GaMMA.git
```

### Step 5: Configure SeisBench

```cmd
# Create cache directory
mkdir %USERPROFILE%\.seisbench

# Set environment variable (in PowerShell)
$env:SEISBENCH_CACHE_ROOT = "$env:USERPROFILE\.seisbench"
```

## HPC/Cluster Installation

Many HPC systems use module systems. Here's a typical workflow:

### Step 1: Load Required Modules

```bash
# Load Python module
module load python/3.10

# Load CUDA if available
module load cuda/11.8

# Check available modules
module avail
```

### Step 2: Create Virtual Environment in Your Home Directory

```bash
# Navigate to your home directory
cd ~

# Create virtual environment
python -m venv seisbench_env

# Activate
source ~/seisbench_env/bin/activate
```

### Step 3: Install Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install with --user flag if needed
pip install numpy scipy pandas matplotlib seaborn obspy pyproj tqdm

# Install PyTorch (check CUDA version on your cluster)
pip install torch torchvision torchaudio

# Install SeisBench and GAMMA
pip install git+https://github.com/seisbench/seisbench
pip install git+https://github.com/AI4EPS/GaMMA.git
```

### Step 4: Create Job Submission Script

Create a file `run_picker.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=picker_aggregator
#SBATCH --output=picker_%j.out
#SBATCH --error=picker_%j.err
#SBATCH --time=48:00:00 #usually enough for two days of data processing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1  # Request 1 GPU if available

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/seisbench_env/bin/activate

# Set cache directory
export SEISBENCH_CACHE_ROOT=~/.seisbench

# Run the script
python picker-aggregator.py
```

Submit with: `sbatch run_picker.sh`

## GPU Setup

### Check GPU Availability

**Linux/Mac:**
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check CUDA version
nvcc --version
```

**Windows:**
```cmd
nvidia-smi
```

### Install CUDA Toolkit

1. Visit [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Download appropriate version (11.8 recommended)
3. Follow installation instructions for your OS

### Install GPU-Enabled PyTorch

After installing CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Setup

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Verification

### Test Installation

Create a test script `test_installation.py`:

```python
#!/usr/bin/env python
"""Test script to verify all dependencies are installed correctly."""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'obspy': 'ObsPy',
        'pyproj': 'PyProj',
        'torch': 'PyTorch',
        'seisbench': 'SeisBench',
        'gamma': 'GAMMA'
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_gpu():
    """Test GPU availability."""
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")

def test_seisbench_models():
    """Test if SeisBench models can be loaded."""
    import seisbench.models as sbm
    
    try:
        model = sbm.PhaseNet.from_pretrained("instance")
        print("✓ SeisBench models can be loaded")
        return True
    except Exception as e:
        print(f"✗ Error loading SeisBench models: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Picker Aggregator Installation")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing package imports...")
    imports_ok = test_imports()
    
    # Test GPU
    print("\n2. Testing GPU support...")
    test_gpu()
    
    # Test SeisBench
    print("\n3. Testing SeisBench models...")
    models_ok = test_seisbench_models()
    
    # Summary
    print("\n" + "=" * 60)
    if imports_ok and models_ok:
        print("✓ All tests passed! Installation successful.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check errors above.")
        sys.exit(1)
```

Run the test:
```bash
python test_installation.py
```

## Troubleshooting

### Common Issues

#### "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
source ~/.venv/seisbench/bin/activate  # Linux/Mac
# OR
seisbench_env\Scripts\activate  # Windows

# Reinstall the missing package
pip install <package_name>
```

#### PyTorch GPU not working

**Solution:**
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Permission errors on Linux/Mac

**Solution:**
```bash
# Don't use sudo with pip
# Instead, use virtual environment or --user flag
pip install --user <package_name>
```

#### SeisBench cache errors

**Solution:**
```bash
# Create cache directory
mkdir -p ~/.seisbench

# Set environment variable
export SEISBENCH_CACHE_ROOT=~/.seisbench

# Or in the Python script, add:
# import seisbench
# seisbench.cache_root = "/path/to/cache"
```

#### Out of memory errors

**Solution:**
- Reduce `BATCH_SIZE` in configuration
- Reduce `PARALLEL_WORKERS`
- Process shorter time periods
- Add more RAM or use a machine with more memory

### Getting Additional Help

1. Check the [SeisBench documentation](https://seisbench.readthedocs.io/)
2. Check the [GAMMA documentation](https://github.com/AI4EPS/GaMMA)
3. Review ObsPy documentation for data access issues
4. Open an issue on the repository

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Review the configuration parameters in `picker-aggregator.py`
3. Test with a small date range first
4. Review the example output

## System-Specific Notes

### macOS Apple Silicon (M1/M2/M3)

PyTorch has MPS (Metal Performance Shaders) support:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# The code will automatically detect and use MPS if available
```

### Windows WSL (Windows Subsystem for Linux)

Follow the Linux installation instructions in WSL. For GPU support, you need WSL2 with CUDA support:

1. Install WSL2
2. Install NVIDIA CUDA on WSL2
3. Follow Linux GPU setup instructions

### Docker Installation (Advanced)

For a containerized environment:

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install numpy scipy pandas matplotlib seaborn obspy pyproj tqdm torch
RUN pip install git+https://github.com/seisbench/seisbench
RUN pip install git+https://github.com/AI4EPS/GaMMA.git

COPY picker-aggregator.py .

CMD ["python", "picker-aggregator.py"]
```
