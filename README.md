# Pick Aggregator: Multi-Model Earthquake Detection System

**Developed by:** Asiye Aziz Zanjani, 2025

## Citation

If you use this software, please cite:

```
Pick Aggregator: A Multi-Model Ensemble of Machine Learning Pickers to Generate an Onshore-Offshore Seismic Catalog for Puerto Rico and the Virgin Islands, Asiye Aziz Zanjani and Heather R. DeShon, SRL (2026)
```

## Overview

Pick Aggregator is a comprehensive Python tool that combines multiple machine learning models to detect and catalog seismic events. It uses an ensemble approach, aggregating picks from different seismic phase detection models to improve accuracy and reliability in earthquake detection.

### Key Features

- **Multi-Model Ensemble**: Combines predictions from multiple state-of-the-art seismic pickers (PhaseNet, OBSTransformer, PickBlue)
- **Network Support**: Handles both ocean bottom seismometers (OBS) and land-based stations
- **Intelligent Aggregation**: Only accepts picks detected by multiple models (configurable threshold)
- **GAMMA Integration**: Uses the GAMMA algorithm for event association and catalog creation
- **Optimized Processing**: Implements data caching, parallel processing, and batch operations for efficiency
- **Flexible Configuration**: Easily customizable parameters for different regions and use cases

## System Requirements

### Hardware
- **GPU Recommended**: CUDA-compatible GPU for faster processing
- **RAM**: Minimum 16GB (32GB+ recommended for large datasets)
- **Storage**: Adequate space for waveform caching and results

### Software
- Python 3.8 or higher
- CUDA toolkit (if using GPU)

## Installation

### Step 1: Create Python Environment

```bash
# Create a virtual environment
python -m venv ~/.venv/seisbench

# Activate the environment
source ~/.venv/seisbench/bin/activate  # On Linux/Mac
# OR
.venv\seisbench\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install pyproj seaborn obspy pandas numpy matplotlib tqdm torch

# Install SeisBench
pip install git+https://github.com/seisbench/seisbench

# Install GAMMA
pip install git+https://github.com/AI4EPS/GaMMA.git
```

### Step 3: Verify Installation

```python
import seisbench
import gamma
print("Installation successful!")
```

## Quick Start Guide

### 1. Basic Configuration

Open `picker-aggregator.py` and modify the user configuration section:

```python
# Time range for analysis
TSTART = "2015-05-15"  # Start date (YYYY-MM-DD)
TEND = "2015-05-29"    # End date (YYYY-MM-DD)

# Network codes
ZZ_NETWORK = "ZZ"  # Ocean bottom network code
PR_NETWORK = "PR"  # Land-based network code

# Aggregation settings
MIN_PICKERS_FOR_AGGREGATION = 2  # Minimum pickers that must agree
TIME_TOLERANCE_SECONDS = 0.5     # Time window for matching picks
```

### 2. Run the Analysis

```bash
# Activate your environment
source ~/.venv/seisbench/bin/activate

# Run the script
python picker-aggregator.py
```

### 3. Monitor Progress

The script will:
1. Download and cache waveform data
2. Run multiple picker models in parallel
3. Aggregate picks from different models
4. Associate events using GAMMA
5. Save results to the output directory

## Understanding the Configuration

### Time and Network Parameters

```python
TSTART = "2015-05-15"  # Analysis start date
TEND = "2015-05-29"    # Analysis end date
ZZ_NETWORK = "ZZ"      # Network code for ocean bottom seismometers
PR_NETWORK = "PR"      # Network code for land stations
CHANNELS = "HH?,BH?"   # Channel patterns to download
TIME_OVERLAP = 20      # Overlap in seconds between time windows
```

### Picker Configuration

The script uses different pickers optimized for each network type:

**Ocean Bottom Seismometers (ZZ Network):**
```python
ZZ_PICKERS = {
    "OBSTransformer_obst2024": {"model": "OBSTransformer", "pretrained": "obst2024"},
    "PickBlue_phasenet": {"model": "PickBlue", "pretrained": "phasenet"}
}
```

**Land Stations (PR Network):**
```python
PR_PICKERS = {
    "PhaseNet_instance": {"model": "PhaseNet", "pretrained": "instance"},
    "PhaseNet_stead": {"model": "PhaseNet", "pretrained": "stead"},
    # ... additional PhaseNet variants
}
```

### Aggregation Parameters

```python
MIN_PICKERS_FOR_AGGREGATION = 2  # At least 2 pickers must detect the same pick
TIME_TOLERANCE_SECONDS = 0.5     # Picks within 0.5 seconds are considered the same
```

**Example:** If 3 pickers detect a P-wave arrival at a station within 0.5 seconds of each other, these are combined into a single, more reliable aggregated pick.

### GAMMA Configuration

GAMMA parameters control event association:

```python
GAMMA_CONFIG = {
    "x(km)": (250, 1500),      # Search area in East-West direction
    "y(km)": (1500, 2500),     # Search area in North-South direction
    "z(km)": (0, 100),         # Depth range
    "vel": {"p": 7, "s": 4},   # P and S wave velocities (km/s)
    "min_picks_per_eq": 8,     # Minimum picks to define an event
    "dbscan_eps": 20,          # Time clustering parameter (seconds)
}
```

### Performance Tuning

```python
DOWNLOAD_CHUNK_HOURS = 1    # Download data in 1-hour chunks
BATCH_SIZE = 100            # Process 100 traces at a time
PARALLEL_WORKERS = 4        # Use 4 parallel workers
DATA_CACHE_DIR = "cached_waveforms"  # Where to store downloaded data
```

## Output Structure

After running, your output directory will look like:

```
PICKER_COMPARISON/
├── aggregated-catalog/
│   ├── picks.csv          # Aggregated picks with event assignments
│   └── catalog.csv        # Final earthquake catalog
├── OBSTransformer_obst2024+PhaseNet_instance/
│   ├── picks_[timestamp].csv
│   └── catalog_[timestamp].csv
├── [other picker combinations]/
├── processing_summary.txt
└── cached_waveforms/      # Cached seismic data
```

### Understanding the Output Files

#### `catalog.csv` - Earthquake Catalog
Contains detected earthquakes with:
- **event_idx**: Associated event number
- **time**: Origin time of the earthquake
- **latitude, longitude**: Location coordinates
- **z(km)**: Depth in kilometers
- **magnitude**: Event magnitude
- **num_picks**: Total number of associated picks
- **gamma_score**: Quality metric from GAMMA

#### `picks.csv` - Phase Picks
Contains individual P and S wave detections:
- **station**: Station code
- **timestamp**: Pick time
- **type**: Phase type (p or s)
- **prob**: Detection probability
- **picker**: Which model(s) detected it
- **event_idx**: Associated event number (use this column to associate picks with events)

## Adapting for Your Region

To use this code for a different study area:

### 1. Update Geographic Parameters

```python
# Change to your region's UTM zone
LOCAL_CRS_EPSG = 32619  # Current: UTM zone 19N (Puerto Rico)
UTM_ZONE = 19

# Update GAMMA search area (in km)
GAMMA_CONFIG = {
    "x(km)": (min_x, max_x),  # Your area in East-West
    "y(km)": (min_y, max_y),  # Your area in North-South
    "z(km)": (0, max_depth),  # Appropriate depth range
}
```

### 2. Update Network Codes

```python
ZZ_NETWORK = "YOUR_OBS_NETWORK"   # Your ocean bottom network
PR_NETWORK = "YOUR_LAND_NETWORK"  # Your land-based network
```

### 3. Adjust Velocity Model

```python
GAMMA_CONFIG = {
    "vel": {"p": your_p_velocity, "s": your_s_velocity},
    # Use regional velocity models if available
}
```

### 4. Select Appropriate Pickers

Keep pickers that work well for your data type and seismic setting.

## Troubleshooting

### Common Issues

**Issue: "No data downloaded"**
- Check network codes are correct
- Verify date range has available data
- Confirm station names exist in the network
- Check FDSN client access

**Issue: "Out of memory errors"**
- Reduce `BATCH_SIZE`
- Reduce `PARALLEL_WORKERS`
- Process shorter time periods
- Use smaller time chunks

**Issue: "CUDA out of memory"**
- Process fewer pickers simultaneously
- Reduce batch size
- Set `PARALLEL_WORKERS = 1`

**Issue: "No events found"**
- Check if picks were generated (look at processing summary)
- Lower detection thresholds (`P_THRESHOLD`, `S_THRESHOLD`)
- Reduce `MIN_PICKERS_FOR_AGGREGATION` to 1
- Adjust GAMMA parameters

### Getting Help

1. Check the processing summary file for diagnostic information
2. Review individual picker outputs before aggregation
3. Verify station coordinates are correct
4. Test with a shorter time period first

## Performance Optimization Tips

1. **Use GPU**: Dramatically speeds up picker models
2. **Cache Data**: Run once to download and cache, then reuse
3. **Parallel Processing**: Adjust `PARALLEL_WORKERS` based on your CPU
4. **Batch Processing**: Larger batches are more efficient (if RAM allows)

## Advanced Usage

### Running Specific Picker Combinations Only

Modify the `best_combinations` list in the main function to process only specific pairs:

```python
best_combinations = [
    ("OBSTransformer_obst2024", "PhaseNet_instance"),
    # Add only the combinations you want
]
```

### Changing Aggregation Strategy

Adjust these parameters to change how picks are combined:

```python
MIN_PICKERS_FOR_AGGREGATION = 3  # More conservative (higher confidence)
TIME_TOLERANCE_SECONDS = 3.0     # Stricter time matching
```

## Scientific Background

### Why Multiple Pickers?

Different machine learning models have different strengths:
- **OBSTransformer**: Optimized for ocean bottom seismometer noise
- **PhaseNet**: Excellent for continental stations
- **PickBlue**: Robust across different data types

By aggregating their predictions, we:
1. Reduce false detections
2. Improve pick accuracy
3. Increase confidence in true events

### The Aggregation Process

1. Each picker independently detects P and S waves
2. Picks at the same station within `TIME_TOLERANCE_SECONDS` are clustered
3. Only clusters with ≥ `MIN_PICKERS_FOR_AGGREGATION` picks are kept
4. Average time and probability are calculated
5. GAMMA associates picks into earthquake events

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please contact:
- Asiye Aziz Zanjani:asiye.azizzanjani@nmt.edu or asiye.azizzanjani@gmail.com

## Acknowledgments

This software uses:
- SeisBench (https://github.com/seisbench/seisbench)
- GAMMA (https://github.com/AI4EPS/GaMMA)
- ObsPy (https://docs.obspy.org/)
- PhaseNet (https://github.com/AI4EPS/PhaseNet)

## Version History

- **v1.0** (2025): Initial release
  - Multi-model picker aggregation
  - GAMMA integration
  - Optimized parallel processing
