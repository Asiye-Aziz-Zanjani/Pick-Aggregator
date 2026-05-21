# Code Structure and Explanation

This document provides a detailed explanation of how the Picker Aggregator code works, making it easier for new users and developers to understand and modify the system.

## Table of Contents
1. [Overview](#overview)
2. [Code Architecture](#code-architecture)
3. [Key Functions Explained](#key-functions-explained)
4. [Data Flow](#data-flow)
5. [Configuration Parameters](#configuration-parameters)
6. [Extending the Code](#extending-the-code)

## Overview

The Picker Aggregator follows a pipeline architecture with five main stages:

```
┌─────────────────┐
│ 1. Download     │ → Downloads seismic data from FDSN servers
│    & Cache      │   and stores it locally for reuse
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. Get Station  │ → Retrieves station metadata (location,
│    Information  │   elevation, network codes)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. Run Pickers  │ → Processes data with multiple ML models
│    in Parallel  │   to detect P and S wave arrivals
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. Aggregate    │ → Combines picks from different models
│    Picks        │   that agree on timing
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. Associate    │ → Uses GAMMA to group picks into
│    Events       │   earthquake events and create catalog
└─────────────────┘
```

## Code Architecture

### File Structure

```python
picker-aggregator.py
├── Configuration Section (lines 1-150)
│   └── All user-modifiable parameters
├── Import Section (lines 152-180)
│   └── Required libraries and packages
├── Global Setup (lines 182-195)
│   └── Coordinate systems and client initialization
├── Core Functions (lines 197-650)
│   ├── Data download and caching
│   ├── Preprocessing
│   ├── Picker initialization
│   ├── Pick processing
│   ├── Pick aggregation
│   └── Event association
└── Main Execution (lines 652-750)
    └── Orchestrates the entire workflow
```

## Key Functions Explained

### 1. `download_and_cache_data()`

**Purpose:** Downloads seismic waveform data and stores it locally to avoid repeated downloads.

**How it works:**
```python
def download_and_cache_data(tstart, tend, networks, channels, time_overlap):
    # Converts date strings to UTCDateTime objects
    start_date = UTCDateTime(f"{tstart}T00:00:00")
    end_date = UTCDateTime(f"{tend}T01:00:00")
    
    # Loops through time in hourly chunks
    for each hour in time range:
        # Checks if data already cached
        if cache_file_exists:
            load_from_cache()
        else:
            # Downloads from FDSN server (IRIS)
            download_waveforms()
            # Preprocesses (merge, detrend, taper)
            preprocess_stream()
            # Saves to disk
            save_to_cache()
```

**Key concepts:**
- **Time overlap:** Adds buffer time before/after each window to avoid edge effects
- **Caching:** Stores data as Python pickle files for fast reloading
- **Preprocessing:** Cleans data before picking to improve accuracy

**Customization:**
```python
# Change download chunk size (hours)
DOWNLOAD_CHUNK_HOURS = 1  # Smaller = less memory, slower
                          # Larger = more memory, faster

# Change cache directory
DATA_CACHE_DIR = "my_cache_folder"
```

---

### 2. `runstations()`

**Purpose:** Retrieves station information including geographic coordinates.

**How it works:**
```python
def runstations(tstart, tend, networks, channels):
    # Gets station metadata from FDSN
    inventory = client.get_stations(...)
    
    # Extracts relevant information
    for each station:
        station_data = {
            "id": "NETWORK.STATION.",
            "longitude": station.longitude,
            "latitude": station.latitude,
            "elevation(m)": station.elevation
        }
        
        # Converts to local UTM coordinates
        # (required by GAMMA for distance calculations)
        x_utm, y_utm = convert_to_utm(lon, lat)
```

**Why UTM coordinates?**
- GAMMA needs distances in kilometers
- UTM provides accurate distances in a local area
- Easier than calculating distances on a sphere

**Customization:**
```python
# Exclude problematic stations
EXCLUDED_STATIONS = ['BADSTATION1', 'BADSTATION2']

# Change coordinate system for your region
LOCAL_CRS_EPSG = 32619  # UTM zone 19N (Puerto Rico)
                        # Find your zone: https://epsg.io/
```

---

### 3. `initialize_picker()`

**Purpose:** Loads a pre-trained machine learning model for phase picking.

**How it works:**
```python
def initialize_picker(picker_config):
    model_name = picker_config["model"]      # e.g., "PhaseNet"
    pretrained = picker_config["pretrained"]  # e.g., "instance"
    
    # Loads the appropriate model
    if model_name == "PhaseNet":
        picker = sbm.PhaseNet.from_pretrained(pretrained)
    elif model_name == "OBSTransformer":
        picker = sbm.OBSTransformer.from_pretrained(pretrained)
    
    # Moves model to GPU if available (huge speedup!)
    if torch.cuda.is_available():
        picker.cuda()
    
    return picker
```

**Available models and their strengths:**

| Model | Pretrained Version | Best For |
|-------|-------------------|----------|
| PhaseNet | instance | General continental data |
| PhaseNet | stead | Diverse global data |
| PhaseNet | scedc | Southern California |
| OBSTransformer | obst2024 | Ocean bottom seismometers |
| PickBlue | phasenet | Robust across data types |

**Customization:**
```python
# Add a new picker
ZZ_PICKERS = {
    "MyNewPicker": {"model": "PhaseNet", "pretrained": "newversion"}
}
```

---

### 4. `run_picks_on_cached_data()`

**Purpose:** Runs a picker on cached waveform data to detect P and S waves.

**How it works:**
```python
def run_picks_on_cached_data(cached_data, network, picker, picker_name):
    all_picks = []
    
    # Processes each cached time window
    for time_window, stream in cached_data.items():
        # Filters for specific network (ZZ or PR)
        network_stream = stream.select(network=network)
        
        # Processes in batches to manage memory
        for batch in divide_into_batches(network_stream):
            # Runs the ML model
            picks = picker.classify(
                batch,
                P_threshold=0.15,  # Minimum probability for P wave
                S_threshold=0.15   # Minimum probability for S wave
            )
            
            # Stores pick information
            for pick in picks:
                save_pick_info(pick)
```

**Understanding thresholds:**
```python
P_THRESHOLD = 0.15  # Lower = more picks (more false positives)
                   # Higher = fewer picks (miss real events)
S_THRESHOLD = 0.15  # S waves are harder to detect
                   # Often kept lower than P threshold
```

**Memory management:**
```python
BATCH_SIZE = 100  # Number of traces processed together
                  # Reduce if getting memory errors
                  # Increase for faster processing (if RAM allows)
```

---

### 5. `aggregate_picks()`

**Purpose:** Combines picks from multiple models that detected the same seismic phase.

**How it works:**
```python
def aggregate_picks(all_picks_data, min_pickers=MIN_PICKERS_FOR_AGGREGATION,
                   time_tolerance=TIME_TOLERANCE_SECONDS,
                   method=AGGREGATION_METHOD):
    """
    Aggregate picks detected by multiple pickers.

    This function combines picks from different models that detected the
    same seismic phase. Only picks detected by at least min_pickers models
    within time_tolerance seconds are kept.

    Algorithm:
        1. Group picks by station and phase type
        2. Find clusters of picks within time_tolerance
        3. Keep only clusters with >= min_pickers detections
        4. Compute a representative timestamp and probability using *method*

    Args:
        all_picks_data (dict): Dictionary of {picker_name: picks_dataframe}
        min_pickers (int): Minimum number of pickers required
        time_tolerance (float): Time window in seconds
        method (str): Aggregation method – one of:
            "mean"          Simple arithmetic mean (default / original).
            "highest_prob"  Use the values from the highest-probability pick;
                            other picks in the cluster still count towards
                            min_pickers but do not shift the reported value.
            "weighted_mean" Probability-weighted mean; more confident picks
                            have greater influence on the reported timestamp
                            and probability.

    Returns:
        pandas.DataFrame: Aggregated picks with additional columns:
            - contributing_pickers: Comma-separated list of pickers
            - num_contributing_pickers: Count of pickers that detected it
            - aggregation_method: The method used (for traceability)

    """
    valid_methods = {"mean", "highest_prob", "weighted_mean"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Choose from: {sorted(valid_methods)}"
        )
```

**Example scenario:**
```
        If 3 pickers detect a P-wave at station ABC within 2 seconds:
        - Pick 1: 12:34:56.123 (prob 0.85)
        - Pick 2: 12:34:56.456 (prob 0.92)
        - Pick 3: 12:34:59.789 (prob 0.78)

        "mean"          → timestamp 12:34:57.456,  prob 0.850
        "highest_prob"  → timestamp 12:34:56.456,  prob 0.920  (Pick 2)
        "weighted_mean" → timestamp ~12:34:56.700, prob ~0.857  (skewed toward Pick 2)
```

**Tuning aggregation:**
```python
# More conservative (higher confidence, fewer picks)
MIN_PICKERS_FOR_AGGREGATION = 3
TIME_TOLERANCE_SECONDS = 3.0

# More liberal (lower confidence, more picks)
MIN_PICKERS_FOR_AGGREGATION = 2
TIME_TOLERANCE_SECONDS = 8.0
```

---

### 6. `setup_gamma_config()`

**Purpose:** Configures GAMMA parameters for event association.

**GAMMA's job:** Given a bunch of P and S wave picks, figure out which ones belong to the same earthquake.

**How it works:**
```python
def setup_gamma_config():
    config = {
        # Search region (in km from origin)
        "x(km)": (250, 1500),    # East-West extent
        "y(km)": (1500, 2500),   # North-South extent
        "z(km)": (0, 100),       # Depth range
        
        # Wave velocities (km/s)
        "vel": {
            "p": 7.0,        # P-wave velocity
            "s": 4.0         # S-wave velocity (p/1.75)
        },
        
        # Clustering parameters
        "dbscan_eps": 20,           # Time window (seconds)
        "dbscan_min_samples": 3,    # Minimum picks to form cluster
        "min_picks_per_eq": 8,      # Minimum picks for valid event
        
        # Quality thresholds
        "max_sigma11": 2.0,   # Maximum uncertainty in space
        "max_sigma22": 2.0,   # Maximum uncertainty in time
    }
```

**GAMMA algorithm overview:**
```
1. For each P pick:
   - Calculate expected S arrival times at all stations
   - Based on assumed earthquake location and velocities
   
2. Find matches:
   - Look for actual S picks that match predictions
   - Within time tolerance (dbscan_eps)
   
3. Cluster picks:
   - Group picks that fit the same earthquake model
   - Require minimum number of picks (min_picks_per_eq)
   
4. Locate events:
   - Optimize earthquake location using all associated picks
   - Calculate uncertainties
```

**Customization for your region:**
```python
GAMMA_CONFIG = {
    # Adjust search area to cover your seismic zone
    "x(km)": (min_x, max_x),  # From station coordinates
    "y(km)": (min_y, max_y),
    
    # Use regional velocity model if available
    "vel": {"p": 6.5, "s": 3.7},  # Example: slower crust
    
    # Adjust quality requirements
    "min_picks_per_eq": 6,  # Lower for sparse networks
                            # Higher for dense networks
}
```

---

### 7. `runevents()`

**Purpose:** Associates picks into earthquakes using GAMMA and converts coordinates.

**How it works:**
```python
def runevents(picks, stations, output_dir, picker_combination):
    # Prepares configuration
    config = setup_gamma_config()
    
    # Runs GAMMA association
    catalogs, assignments = association(picks, stations, config)
    
    # Processes results
    for each earthquake in catalogs:
        # Converts UTM back to lat/lon
        lat, lon = convert_utm_to_latlon(x_km, y_km)
        
        # Saves event information
        save_event(time, lat, lon, depth, magnitude, ...)
    
    # Saves pick-event associations
    for each pick:
        save_assignment(pick_id, event_id, probability)
```

**Output files:**
```
catalog.csv:
  event_index | time      | latitude | longitude | depth | magnitude | num_picks
  0           | 2015-... | 18.234   | -66.123   | 15.3  | 2.4       | 12
  1           | 2015-... | 18.456   | -66.456   | 8.7   | 1.8       | 9

picks.csv:
  pick_idx | event_idx | station | timestamp | type | prob | picker
  0        | 0         | ABC     | 2015-...  | p    | 0.92 | AGGREGATED
  1        | 0         | DEF     | 2015-...  | s    | 0.85 | AGGREGATED
```

---

### 8. `process_picker_parallel()`

**Purpose:** Wrapper function to run a single picker in parallel with others.

**How it works:**
```python
def process_picker_parallel(args):
    picker_name, picker_config, network, cached_data = args
    
    # Initializes the picker model
    picker = initialize_picker(picker_config)
    
    # Runs picking on all cached data
    picks = run_picks_on_cached_data(cached_data, network, picker, picker_name)
    
    # Cleans up memory (important for parallel processing!)
    del picker
    torch.cuda.empty_cache()
    
    return picker_name, picks
```

**Parallel processing concept:**
```
Traditional (Sequential):
  Picker 1 → [======] 10 minutes
  Picker 2 → [======] 10 minutes  
  Picker 3 → [======] 10 minutes
  Total: 30 minutes

Parallel (with 3 workers):
  Picker 1 → [======]
  Picker 2 → [======] } 10 minutes
  Picker 3 → [======]
  Total: 10 minutes
```

**Configuration:**
```python
PARALLEL_WORKERS = 4  # Number of pickers running simultaneously
                      # Set to number of CPU cores available
                      # Reduce if running out of memory
```

---

## Data Flow

### Complete Pipeline Example

```
INPUT: Time range 2015-05-15 to 2015-05-29

Step 1: Download Data
  ├─ Hour 1: 2015-05-15 00:00 - 01:00 → 150 traces
  ├─ Hour 2: 2015-05-15 01:00 - 02:00 → 152 traces
  └─ ... (continuing for 14 days)

Step 2: Station Info
  ├─ Network ZZ: 15 ocean bottom stations
  └─ Network PR: 28 land stations
  Total: 43 stations with coordinates

Step 3: Run Pickers (Parallel)
  ├─ OBSTransformer on ZZ data → 1,234 P picks, 892 S picks
  ├─ PickBlue on ZZ data → 1,156 P picks, 823 S picks
  ├─ PhaseNet (instance) on PR → 3,456 P picks, 2,987 S picks
  ├─ PhaseNet (stead) on PR → 3,389 P picks, 2,876 S picks
  └─ ... (other PhaseNet variants)
  Total: ~25,000 picks

Step 4: Aggregate Picks
  ├─ Group by station and phase
  ├─ Cluster picks within 5 seconds
  ├─ Keep clusters with ≥2 pickers
  └─ Result: 8,456 aggregated picks

Step 5: Associate Events
  ├─ GAMMA processes aggregated picks
  ├─ Finds time-space clusters
  ├─ Requires ≥8 picks per event
  └─ Result: 147 earthquakes

OUTPUT: Earthquake catalog with locations and magnitudes
```

## Configuration Parameters

### Critical Parameters to Adjust

#### 1. Geographic Extent
```python
# Match to your study area
LOCAL_CRS_EPSG = 32619  # Find at epsg.io
GAMMA_CONFIG = {
    "x(km)": (xmin, xmax),  # From station locations
    "y(km)": (ymin, ymax),
    "z(km)": (0, max_depth)
}
```

#### 2. Network Codes
```python
ZZ_NETWORK = "YOUR_NETWORK_CODE"
PR_NETWORK = "YOUR_OTHER_NETWORK"
```

#### 3. Quality Control
```python
# Detection sensitivity
P_THRESHOLD = 0.15  # Lower = more sensitive
S_THRESHOLD = 0.15

# Aggregation requirements
MIN_PICKERS_FOR_AGGREGATION = 2  # Higher = more confident
TIME_TOLERANCE_SECONDS = 5.0      # Larger = more grouping

# Event requirements
GAMMA_CONFIG["min_picks_per_eq"] = 8  # Minimum picks per earthquake
```

#### 4. Performance Tuning
```python
BATCH_SIZE = 100          # Traces per batch
PARALLEL_WORKERS = 4      # Simultaneous pickers
DOWNLOAD_CHUNK_HOURS = 1  # Data download size
```

## Extending the Code

### Adding a New Picker Model

```python
# 1. Add to configuration
ZZ_PICKERS = {
    "MyNewPicker": {
        "model": "ModelName",
        "pretrained": "version"
    }
}

# 2. Update initialize_picker if needed
def initialize_picker(picker_config):
    # ... existing code ...
    elif model_name == "ModelName":
        picker = sbm.ModelName.from_pretrained(pretrained)
    return picker
```

### Adding Custom Preprocessing

```python
def preprocess_stream(stream):
    """Add custom preprocessing steps."""
    # Existing preprocessing
    stream = stream.merge(fill_value=0)
    stream = Stream([tr for tr in stream if tr.stats.npts > 100])
    
    # Add your custom steps
    for tr in stream:
        tr.detrend('demean')
        tr.taper(max_percentage=0.05)
        
        # Example: Apply bandpass filter
        tr.filter('bandpass', freqmin=1.0, freqmax=10.0)
    
    return stream
```

### Modifying Aggregation Logic

```python
def aggregate_picks(all_picks_data, min_pickers=2, time_tolerance=5.0):
    """Modify to use weighted averaging."""
    
    # ... existing clustering code ...
    
    # Instead of simple average:
    # avg_time = mean(cluster_times)
    
    # Use weighted average by probability:
    weights = [pick['prob'] for pick in cluster_picks]
    times = [pick['timestamp'] for pick in cluster_picks]
    avg_time = np.average(times, weights=weights)
    
    # ... rest of code ...
```

### Custom Output Formats

```python
def save_custom_format(catalog, output_file):
    """Save catalog in a custom format."""
    import json
    
    events = []
    for idx, event in catalog.iterrows():
        events.append({
            'id': int(event['event_index']),
            'time': event['time'].isoformat(),
            'location': {
                'lat': float(event['latitude']),
                'lon': float(event['longitude']),
                'depth_km': float(event['z(km)'])
            },
            'magnitude': float(event['magnitude']),
            'quality': {
                'num_picks': int(event['num_picks']),
                'gamma_score': float(event['gamma_score'])
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)
```

## Best Practices

### For Beginners
1. Start with a short time period (1-2 days)
2. Use default thresholds initially
3. Check outputs at each stage
4. Gradually increase time range

### For Advanced Users
1. Tune thresholds based on your network geometry
2. Add custom pickers trained on your data
3. Implement quality control filters
4. Optimize parallel processing for your hardware

### Performance Optimization
1. Use GPU if available (10-50x speedup)
2. Cache data for repeated runs
3. Adjust batch size based on RAM
4. Use multiple workers but monitor memory

### Debugging Tips
1. Check `processing_summary.txt` for overview
2. Examine individual picker outputs before aggregation
3. Verify station coordinates are correct
4. Test GAMMA parameters with known events
5. Use smaller datasets for testing

## Common Modifications

### Processing a Different Region

```python
# 1. Change network codes
ZZ_NETWORK = "MY_NETWORK"

# 2. Update coordinate system
LOCAL_CRS_EPSG = 32610  # Example: UTM Zone 10N

# 3. Adjust GAMMA search area
GAMMA_CONFIG = {
    "x(km)": (100, 500),     # Based on your stations
    "y(km)": (200, 600),
    "z(km)": (0, 50),        # If shallow earthquakes
}

# 4. Update velocity model
GAMMA_CONFIG["vel"] = {"p": 6.0, "s": 3.5}  # Your region
```

### Processing Only Specific Stations

```python
# Add station whitelist
INCLUDED_STATIONS = ['STA1', 'STA2', 'STA3']

def runstations(tstart, tend, networks, channels):
    # ... existing code ...
    if station.code in INCLUDED_STATIONS:  # Add this check
        stations.append({...})
```

### Saving Additional Outputs

```python
def runevents(picks, stations, output_dir, picker_combination):
    # ... existing code ...
    
    # Save additional diagnostic info
    diagnostic_file = os.path.join(output_dir, "diagnostics.txt")
    with open(diagnostic_file, 'w') as f:
        f.write(f"Total picks: {len(picks)}\n")
        f.write(f"Total events: {len(catalog)}\n")
        f.write(f"Picks per event: {len(picks)/len(catalog):.1f}\n")
```

This should help new users understand the code structure and make informed modifications!
