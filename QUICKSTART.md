# Quick Start Guide

Get up and running with Picker Aggregator in 15 minutes!

## Prerequisites

- Linux, macOS, or Windows 10+
- Python 3.8 or higher
- 16GB RAM minimum
- Stable internet connection

## Installation

### 1. Create and activate virtual environment

**Linux/Mac:**
```bash
python3 -m venv ~/.venv/seisbench
source ~/.venv/seisbench/bin/activate
```

**Windows:**
```cmd
python -m venv seisbench_env
seisbench_env\Scripts\activate
```

### 2. Install packages

```bash
pip install numpy pandas matplotlib seaborn obspy pyproj tqdm torch
pip install git+https://github.com/seisbench/seisbench
pip install git+https://github.com/AI4EPS/GaMMA.git
```

### 3. Download the code

```bash
git clone https://github.com/yourusername/picker-aggregator.git
cd picker-aggregator
```

## First Run 

### 1. Test with a short time period

Open `picker-aggregator.py` and modify these lines:

```python
# Use a short test period (2 days instead of 14)
TSTART = "2015-05-15"
TEND = "2015-05-17"  # Changed from 2015-05-29

# Use fewer pickers for faster testing
PR_PICKERS = {
    "PhaseNet_instance": {"model": "PhaseNet", "pretrained": "instance"},
    # Comment out the others for now
}
```

### 2. Run the script

```bash
python picker-aggregator.py
```

You should see output like:
```
=== Starting Optimized Multi-Picker Seismic Analysis ===

STEP 1: Downloading and caching data
Downloading and caching data from 2015-05-15 to 2015-05-17
Downloaded and cached 156 traces for 2015-05-15 00:00:00 to 2015-05-15 01:00:00
...

STEP 2: Getting station information
Total stations found: 43

STEP 3: Processing pickers in parallel
Starting OBSTransformer_obst2024 for network ZZ
...
```

### 3. Check the results

After completion (5-10 minutes), check the output:

```bash
cd PICKER_COMPARISON/aggregated-catalog/
ls
```

You should see:
- `picks.csv` - Detected seismic phases
- `catalog.csv` - Earthquake catalog

## Understanding Your Results

### View the earthquake catalog

```bash
# On Linux/Mac
head -20 catalog.csv

# Or open in any spreadsheet program
```

**Key columns:**
- `time`: When the earthquake occurred
- `latitude`, `longitude`: Location
- `z(km)`: Depth in kilometers
- `magnitude`: Event size
- `num_picks`: Number of seismic phases detected

### View the picks

```bash
head -20 picks.csv
```

**Key columns:**
- `station`: Which seismometer detected it
- `timestamp`: When it was detected
- `type`: P-wave or S-wave
- `prob`: Confidence (0-1)
- `event_idx`: Which earthquake it belongs to

## Common First-Run Issues

### Issue: "No module named 'seisbench'"

**Solution:**
```bash
# Make sure virtual environment is activated
source ~/.venv/seisbench/bin/activate

# Reinstall
pip install git+https://github.com/seisbench/seisbench
```

### Issue: "No data downloaded"

**Solution:**
- Check your internet connection
- Verify the date range has data
- Try a different time period

### Issue: "Out of memory"

**Solution:** Open `picker-aggregator.py` and reduce:
```python
BATCH_SIZE = 50  # Reduced from 100
PARALLEL_WORKERS = 2  # Reduced from 4
```

### Issue: Very slow processing

**Solutions:**
1. Use GPU if available (see INSTALLATION.md)
2. Process shorter time periods
3. Use fewer picker models
4. Increase batch size (if you have RAM)

## Next Steps

### 1. Process your full time period

Once the test works, change back to your desired dates:
```python
TSTART = "2015-05-15"
TEND = "2015-05-29"
```

### 2. Enable all pickers

Uncomment all the pickers in the configuration:
```python
PR_PICKERS = {
    "PhaseNet_instance": {"model": "PhaseNet", "pretrained": "instance"},
    "PhaseNet_stead": {"model": "PhaseNet", "pretrained": "stead"},
    # ... uncomment all
}
```

### 3. Optimize for your system

Adjust these based on your hardware:
```python
# If you have lots of RAM
BATCH_SIZE = 200
PARALLEL_WORKERS = 8

# If you have GPU
# Code automatically uses it, but you can verify with:
import torch
print(torch.cuda.is_available())  # Should print True
```

### 4. Customize for your region

See the full README.md for details on adapting to your study area:
- Change network codes
- Update coordinate system
- Adjust velocity model
- Modify search parameters

## Verification Checklist

After your first successful run:

- `PICKER_COMPARISON/` directory exists
- `aggregated-catalog/` subdirectory exists
- `catalog.csv` contains earthquake events
- `picks.csv` contains phase detections
- `processing_summary.txt` shows no errors
- `cached_waveforms/` contains data files

## Getting Help

If you encounter issues:

1. Check the `processing_summary.txt` file
2. Review the error messages carefully
3. See the [full documentation](README.md)
4. Check [INSTALLATION.md](INSTALLATION.md) for detailed setup
5. Review [CODE_EXPLANATION.md](CODE_EXPLANATION.md) to understand the workflow

## Useful Commands

### Check Python version
```bash
python --version  # Should be 3.8 or higher
```

### Check if packages installed
```bash
pip list | grep seisbench
pip list | grep gamma
```

### Check GPU availability
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Clean up and restart
```bash
# Remove cached data
rm -rf cached_waveforms/

# Remove previous results
rm -rf PICKER_COMPARISON/

# Run again
python picker-aggregator.py
```

## Example Output

A successful run will produce output like:

```
=== PROCESSING SUMMARY ===
Time range: 2015-05-15 to 2015-05-17
Networks: ZZ, PR
Channels: HH?,BH?
Total cached data chunks: 48
Total pickers processed: 8

Picker Results:
  OBSTransformer_obst2024: 1234 picks
  PickBlue_phasenet: 1156 picks
  PhaseNet_instance: 3456 picks
  PhaseNet_stead: 3389 picks
  PhaseNet_original: 3312 picks
  PhaseNet_ceed: 3445 picks
  PhaseNet_crew: 3401 picks
  PhaseNet_scedc: 3298 picks

Aggregated picks: 8456
Events detected: 147
```

## Performance Expectations

Typical processing times (for 2 days of data, 43 stations):

**CPU only:**
- Downloading data: 10-20 minutes
- Running pickers: 30-60 minutes per picker
- Aggregation: 1-2 minutes
- Event association: 2-5 minutes
- **Total: 2-4 hours**

**With GPU:**
- Downloading data: 10-20 minutes
- Running pickers: 3-5 minutes per picker
- Aggregation: 1-2 minutes
- Event association: 2-5 minutes
- **Total: 30-60 minutes**

## Tips for Success

1. **Start small**: Test with 1-2 days before processing weeks/months
2. **Use GPU**: 10-50x speedup for picker models
3. **Monitor resources**: Use `top` (Linux/Mac) or Task Manager (Windows)
4. **Check logs**: Review output messages for warnings
5. **Validate results**: Compare with existing catalogs if available

After your first successful run:

1. **Analyze your catalog**: Plot earthquake locations, magnitudes over time
2. **Tune parameters**: Adjust thresholds based on your network
3. **Process more data**: Extend your time range
4. **Compare catalogs**: See how different picker combinations perform
5. **Share results**: Export to standard formats for publication

## Example Analysis

Quick Python script to visualize your results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load catalog
catalog = pd.read_csv('PICKER_COMPARISON/aggregated-catalog/catalog.csv', 
                     sep='\t')

# Plot earthquake locations
plt.figure(figsize=(10, 8))
plt.scatter(catalog['longitude'], catalog['latitude'], 
           c=catalog['z(km)'], s=catalog['magnitude']*20,
           cmap='viridis', alpha=0.6)
plt.colorbar(label='Depth (km)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Detected Earthquakes')
plt.savefig('earthquake_map.png')
print("Map saved to earthquake_map.png")
```

## Next Steps

Once you're comfortable with the basic workflow:

1. Read the full [README.md](README.md)
2. Review [CODE_EXPLANATION.md](CODE_EXPLANATION.md) for advanced usage
3. Check [INSTALLATION.md](INSTALLATION.md) for HPC setup
4. Customize parameters for your specific region
