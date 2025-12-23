#!/usr/bin/env python3
"""
Picker Aggregator: Multi-model Machine Learning Picker for Integrated Earthquake Catalog

Developed by: Asiye Aziz Zanjani, 2025

Citation:
    Picker Aggregator: A Multi-model Ensembel of Machine Learning Pickers for Seismic Phase
    Detection and Association to Generate Comprehensive Seismic Catalog North of Puerto Rico 
    and Virgin Islands, Asiye Aziz Zanjani and Heather R. DeShon (2025)

Description:
    This script combines multiple machine learning models to detect seismic phases 
    (P and S waves) from seismograms. It downloads waveform data, runs multiple 
    picker models, aggregates their predictions, and creates an earthquake catalog 
    using the GAMMA association algorithm.

Quick Start:
    1. Modify the configuration parameters below (marked with USER CONFIGURATION)
    2. Activate your environment: source ~/.venv/seisbench/bin/activate
    3. Run: python picker-aggregator.py using sbatch on HPC
    4. Results will be in PICKER_COMPARISON/aggregated-catalog/

For detailed documentation, see README.md
"""

# ================================================================================
# USER CONFIGURATION PARAMETERS - MODIFY THESE FOR YOUR STUDY
# ================================================================================

# --------------------------------------------------------------------------------
# TIME RANGE CONFIGURATION
# --------------------------------------------------------------------------------
# Define the time period for analysis (format: YYYY-MM-DD)
TSTART = "2015-05-15"  # Start date
TEND = "2015-05-17"    # End date (inclusive)

# TIP: Start with a short period (2 days) for testing, then expand

# --------------------------------------------------------------------------------
# NETWORK AND CHANNEL CONFIGURATION
# --------------------------------------------------------------------------------
# Network codes for your seismometer networks
ZZ_NETWORK = "ZZ"  # Ocean bottom seismometer network code
PR_NETWORK = "PR"  # Land-based seismometer network code

# Channel patterns to download (wildcards supported)
# HH = High sample rate, broad band
# BH = Broad band
# ? = wildcard for component (Z, N, E, 1, 2, 3)
CHANNELS = "HH?,BH?"

# Time overlap between download windows (seconds)
# Adds buffer before/after each time window to avoid edge effects
TIME_OVERLAP = 20

# --------------------------------------------------------------------------------
# DATA SOURCE CONFIGURATION
# --------------------------------------------------------------------------------
# FDSN data client
CLIENT_NAME = "IRIS"

# --------------------------------------------------------------------------------
# COORDINATE SYSTEM CONFIGURATION
# --------------------------------------------------------------------------------
# IMPORTANT: Adjust these for your study region!
# WGS84 EPSG code (standard latitude/longitude system)
WGS84_EPSG = 4139

# Local UTM coordinate system (for distance calculations)
# Puerto Rico region uses UTM Zone 19N (EPSG:32619)
# Find your UTM zone: https://epsg.io/
LOCAL_CRS_EPSG = 32619

# UTM projection parameters
UTM_ZONE = 19        # UTM zone number for your region
UTM_ELLPS = "WGS84"  # Ellipsoid model
UTM_SOUTH = False    # Set True if in southern hemisphere

# --------------------------------------------------------------------------------
# STATION FILTERING
# --------------------------------------------------------------------------------
# List of stations to exclude from analysis (e.g., known bad stations)
EXCLUDED_STATIONS = ['AUA1']

# TIP: Add stations here if they produce many false picks or have data quality issues

# --------------------------------------------------------------------------------
# PICKER DETECTION THRESHOLDS
# --------------------------------------------------------------------------------
# Probability thresholds for P and S wave detection (range: 0.0 to 1.0)
# Lower values = more sensitive (more picks, including false positives)
# Higher values = more conservative (fewer picks, may miss some events)
P_THRESHOLD = 0.2  # P-wave detection threshold
S_THRESHOLD = 0.2  # S-wave detection threshold

# TIP: Start with 0.1, then adjust based on results
#      - Too many false picks? Increase to 0.2-0.3
#      - Missing obvious events? Decrease to 0.1

# --------------------------------------------------------------------------------
# PICK AGGREGATION PARAMETERS
# --------------------------------------------------------------------------------
# Minimum number of pickers that must agree to accept a pick
# 2 = at least 2 models must detect the same phase
# 3 = more conservative, higher confidence
MIN_PICKERS_FOR_AGGREGATION = 2

# Time window for considering picks as the same event (seconds)
# Picks within this window are grouped together
TIME_TOLERANCE_SECONDS = 5.0

# TIP: Increase TIME_TOLERANCE_SECONDS if your clocks are not well synchronized

# --------------------------------------------------------------------------------
# PERFORMANCE AND MEMORY PARAMETERS
# --------------------------------------------------------------------------------
# Download data in chunks (hours per chunk)
# Smaller = less memory usage, more download requests
# Larger = more memory usage, fewer download requests
DOWNLOAD_CHUNK_HOURS = 1

# Number of traces to process simultaneously
# Reduce if you get out-of-memory errors
# Increase if you have plenty of RAM (speeds up processing)
BATCH_SIZE = 100

# Number of parallel workers for picker processing
# Set to number of CPU cores available
# Reduce if running out of memory
PARALLEL_WORKERS = 4

# Directory for caching downloaded waveform data
# Data is saved here to avoid re-downloading on subsequent runs
DATA_CACHE_DIR = "cached_waveforms"

# --------------------------------------------------------------------------------
# PICKER MODEL CONFIGURATIONS
# --------------------------------------------------------------------------------
# Pickers for Ocean Bottom Seismometer (OBS) network
# These models are optimized for ocean bottom data with higher noise levels
ZZ_PICKERS = {
    "OBSTransformer_obst2024": {
        "model": "OBSTransformer", 
        "pretrained": "obst2024"
    },
    "PickBlue_phasenet": {
        "model": "PickBlue", 
        "pretrained": "phasenet"
    }
}

# Pickers for Land-based seismometer network
# These models are trained on continental/crustal data
PR_PICKERS = {
    "PhaseNet_instance": {
        "model": "PhaseNet", 
        "pretrained": "instance"
    },
    "PhaseNet_stead": {
        "model": "PhaseNet", 
        "pretrained": "stead"
    },
    "PhaseNet_original": {
        "model": "PhaseNet", 
        "pretrained": "original"
    },
    "PhaseNet_ceed": {
        "model": "PhaseNet", 
        "pretrained": "ceed"
    },
    "PhaseNet_scedc": {
        "model": "PhaseNet", 
        "pretrained": "scedc"
    }
}

# TIP: You can comment out pickers to speed up testing:
#      Just add # at the start of lines you want to disable

# --------------------------------------------------------------------------------
# GAMMA CONFIGURATION FOR EVENT ASSOCIATION
# --------------------------------------------------------------------------------
# GAMMA (Gaussian Mixture Model Association) parameters
# These control how picks are grouped into earthquake events
GAMMA_CONFIG = {
    # Spatial dimensions for calculations
    "dims": ['x(km)', 'y(km)', 'z(km)'],
    
    # Use DBSCAN clustering for initial pick grouping
    "use_dbscan": True,
    
    # Use amplitude information (set False if not available)
    "use_amplitude": False,
    
    # Search region boundaries (kilometers)
    # IMPORTANT: Adjust these to cover your seismic zone!
    # Get approximate values from your station coordinates
    "x(km)": (250, 1500),    # East-West extent
    "y(km)": (1500, 2500),   # North-South extent
    "z(km)": (0, 100),       # Depth range (0 = surface, positive = deeper)
    
    # Seismic wave velocities (km/s)
    # IMPORTANT: Use regional velocity model if available!
    "vel": {
        "p": 7,      # P-wave velocity (typical crustal: 5.5-7.0 km/s)
        "s": 7/1.75  # S-wave velocity (typically Vp/1.73 to Vp/1.78)
    },
    
    # Association method
    "method": "BGMM",  # Options: "BGMM" (Bayesian) or "GMM" (standard)
    
    # DBSCAN clustering parameters
    "dbscan_eps": 20,           # Time window for clustering (seconds)
    "dbscan_min_samples": 3,    # Minimum picks to form a cluster
    
    # Event quality requirements
    "min_picks_per_eq": 8,  # Minimum picks required for a valid earthquake
    
    # Location uncertainty thresholds
    "max_sigma11": 2,    # Maximum horizontal uncertainty (km)
    "max_sigma22": 2.0,  # Maximum vertical uncertainty (km)
    "max_sigma12": 2.0   # Maximum covariance uncertainty
}

# TIP: If you're getting too few/many events:
#      - Too few: Decrease min_picks_per_eq, increase dbscan_eps
#      - Too many: Increase min_picks_per_eq, decrease dbscan_eps

# --------------------------------------------------------------------------------
# OUTPUT CONFIGURATION
# --------------------------------------------------------------------------------
# Base directory for all output files
BASE_OUTPUT_DIR = "PICKER_COMPARISON"

# Standard output filenames (don't change unless necessary)
PICKS_FILENAME = "picks.csv"      # Phase pick detections
CATALOG_FILENAME = "catalog.csv"  # Earthquake catalog

# Subdirectory for aggregated results
AGGREGATED_DIR = "aggregated-catalog"

# ================================================================================
# END OF USER CONFIGURATION
# Below this line: Code implementation (modify only if you know what you're doing)
# ================================================================================

# --------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------
# Configure SeisBench cache location before importing
import seisbench
seisbench.cache_root = "/PUT YOUR PATH/.seisbench"  # Adjust to your path

# Standard library imports
import os
import sys
import gc
import pickle
import argparse
import threading as thread
import concurrent.futures
import multiprocessing as mp
from datetime import datetime, timedelta
from collections import Counter

# Scientific computing imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Geospatial and coordinate transformation imports
import pyproj
from pyproj import CRS, Transformer, Proj, transform

# Seismology-specific imports
from obspy import UTCDateTime, Stream, read_events
from obspy.clients.fdsn import Client

# Machine learning and picker imports
import torch
import seisbench.models as sbm
import seisbench.data as sbd
from seisbench.models import PickBlue

# GAMMA association imports
from gamma import BayesianGaussianMixture, GaussianMixture
from gamma.utils import association, convert_picks_csv, from_seconds

# Progress bar
from tqdm import tqdm

# Configure plotting style
sns.set(font_scale=1.2)
sns.set_style("ticks")

# --------------------------------------------------------------------------------
# GLOBAL SETUP
# --------------------------------------------------------------------------------
print("Initializing Picker Aggregator...")
print(f"SeisBench cache: {seisbench.cache_root}")

# Initialize coordinate reference systems for lat/lon to UTM conversion
wgs84 = CRS.from_epsg(WGS84_EPSG)
local_crs = CRS.from_epsg(LOCAL_CRS_EPSG)
transformer = Transformer.from_crs(wgs84, local_crs, always_xy=True)

# Initialize FDSN client for data download
try:
    client = Client(CLIENT_NAME)
    print(f"Connected to {CLIENT_NAME} data center")
except Exception as e:
    print(f"ERROR: Could not connect to {CLIENT_NAME}: {e}")
    sys.exit(1)

# Initialize UTM projections for coordinate conversions
utm_proj = Proj(proj="utm", zone=UTM_ZONE, ellps=UTM_ELLPS, south=UTM_SOUTH)
wgs84_proj = Proj(proj="latlong", ellps=UTM_ELLPS)

# Create cache directory if it doesn't exist
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
print(f"Using cache directory: {DATA_CACHE_DIR}")

# --------------------------------------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------------------------------------

def initialize_picker(picker_config):
    """
    Initialize a machine learning picker model.
    
    Args:
        picker_config (dict): Configuration with 'model' and 'pretrained' keys
        
    Returns:
        Initialized picker model, or None if initialization fails
        
    Example:
        config = {"model": "PhaseNet", "pretrained": "instance"}
        picker = initialize_picker(config)
    """
    model_name = picker_config["model"]
    pretrained = picker_config["pretrained"]
    
    try:
        # Load the appropriate model
        if model_name == "OBSTransformer":
            picker = sbm.OBSTransformer.from_pretrained(pretrained)
        elif model_name == "PhaseNet":
            picker = sbm.PhaseNet.from_pretrained(pretrained)
        elif model_name == "PickBlue":
            picker = PickBlue(pretrained)
        else:
            raise ValueError(f"Unknown picker model: {model_name}")
        
        # Move to GPU if available (provides significant speedup)
        if torch.cuda.is_available():
            print(f"  ✓ Using GPU for {model_name}_{pretrained}")
            picker.cuda()
        else:
            print(f"  ⚠ Using CPU for {model_name}_{pretrained} (GPU not available)")
        
        return picker
        
    except Exception as e:
        print(f"  ✗ Error initializing picker {model_name}_{pretrained}: {e}")
        return None


def setup_gamma_config():
    """
    Setup GAMMA configuration with dynamic parameters.
    
    Returns:
        dict: Complete GAMMA configuration
        
    Note:
        Adds oversample_factor and bfgs_bounds based on method type
    """
    config = GAMMA_CONFIG.copy()
    
    # Set oversample factor based on association method
    if config["method"] == "BGMM":
        config["oversample_factor"] = 4  # Bayesian GMM needs more samples
    elif config["method"] == "GMM":
        config["oversample_factor"] = 1  # Standard GMM
    
    # Set optimization bounds for location search
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x bounds (E-W)
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y bounds (N-S)
        (0, config["z(km)"][1] + 1),                       # z bounds (depth)
        (None, None),                                       # t bounds (time)
    )
    
    return config


def download_and_cache_data(tstart, tend, networks, channels, time_overlap):
    """
    Download seismic waveform data and cache it locally for faster reprocessing.
    
    This function downloads data in hourly chunks and saves them as pickle files.
    On subsequent runs, cached data is loaded instead of re-downloading.
    
    Args:
        tstart (str): Start date (YYYY-MM-DD)
        tend (str): End date (YYYY-MM-DD)
        networks (str): Network codes (comma-separated)
        channels (str): Channel patterns
        time_overlap (int): Seconds of overlap before/after each window
        
    Returns:
        dict: Cached data indexed by time keys
        
    Example:
        cached_data = download_and_cache_data("2015-05-15", "2015-05-17", 
                                             "ZZ,PR", "HH?,BH?", 20)
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: DOWNLOADING AND CACHING DATA")
    print(f"{'='*80}")
    print(f"Time range: {tstart} to {tend}")
    print(f"Networks: {networks}")
    print(f"Channels: {channels}")

    # Convert date strings to ObsPy UTCDateTime objects
    start_date = UTCDateTime(f"{tstart}T00:00:00")
    end_date = UTCDateTime(f"{tend}T01:00:00")
    
    cached_data = {}
    current_time = start_date
    total_hours = int((end_date - start_date) / 3600)
    hour_count = 0
    
    # Loop through each day
    while current_time < end_date:
        # Process 24 hours (one day) at a time
        for n in range(24):
            start_time = current_time + n * 3600
            end_time = current_time + (n + 1) * 3600
            hour_count += 1
            
            # Skip if past end time
            if start_time >= end_date:
                break
            
            # Create unique cache filename based on time window
            cache_filename = os.path.join(
                DATA_CACHE_DIR, 
                f"data_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            
            # Check if data is already cached
            if os.path.exists(cache_filename):
                print(f"[{hour_count}/{total_hours}] Loading cached data for {start_time.strftime('%Y-%m-%d %H:%M')}")
                try:
                    with open(cache_filename, 'rb') as f:
                        cached_data[f"{start_time}_{end_time}"] = pickle.load(f)
                    continue
                except Exception as e:
                    print(f"  ⚠ Error loading cache, will re-download: {e}")
            
            # Download fresh data
            try:
                print(f"[{hour_count}/{total_hours}] Downloading data for {start_time.strftime('%Y-%m-%d %H:%M')}...", end=" ")
                
                # Download waveforms from FDSN server
                all_network = f"{ZZ_NETWORK},{PR_NETWORK}"
                stream = client.get_waveforms(
                    network=all_network, 
                    station="*",           # All stations
                    location="*",          # All locations
                    channel=channels, 
                    starttime=start_time - time_overlap,  # Add buffer before
                    endtime=end_time + time_overlap       # Add buffer after
                )
                
                if len(stream) > 0:
                    # Preprocess the waveform data
                    stream = preprocess_stream(stream)
                    
                    # Save to cache for future use
                    with open(cache_filename, 'wb') as f:
                        pickle.dump(stream, f)
                    
                    # Store in memory
                    cached_data[f"{start_time}_{end_time}"] = stream
                    print(f"✓ {len(stream)} traces")
                else:
                    print("⚠ No data available")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Move to next day
        current_time += timedelta(days=1)
    
    print(f"\nTotal cached time windows: {len(cached_data)}")
    return cached_data


def preprocess_stream(stream):
    """
    Preprocess waveform stream to optimize for phase picking.
    
    Steps:
        1. Merge traces with same ID
        2. Remove very short traces (< 100 samples)
        3. Remove mean (detrend)
        4. Apply taper to avoid edge effects
    
    Args:
        stream (obspy.Stream): Raw waveform data
        
    Returns:
        obspy.Stream: Preprocessed waveform data
    """
    # Merge traces with the same ID (fills gaps with zeros)
    stream = stream.merge(fill_value=0)
    
    # Remove traces that are too short to be useful
    stream = Stream([tr for tr in stream if tr.stats.npts > 100])
    
    # Apply preprocessing to each trace
    for tr in stream:
        # Remove mean (centers data around zero)
        tr.detrend('demean')
        
        # Apply cosine taper to first/last 5% of trace
        # Reduces edge effects in signal processing
        tr.taper(max_percentage=0.05)
    
    return stream


def run_picks_on_cached_data(cached_data, network, picker, picker_name):
    """
    Run seismic phase picking on cached waveform data.
    
    This function processes all cached data with a single picker model,
    detecting P and S wave arrivals at each station.
    
    Args:
        cached_data (dict): Dictionary of cached waveform streams
        network (str): Network code to process (e.g., "ZZ" or "PR")
        picker: Initialized picker model
        picker_name (str): Name of the picker for identification
        
    Returns:
        pandas.DataFrame: Detected picks with columns:
            - id: Full trace ID (network.station.location.channel)
            - network: Network code
            - station: Station code
            - channel: Channel code
            - timestamp: Pick time (datetime)
            - prob: Detection probability (0-1)
            - type: Phase type ('p' or 's')
            - picker: Picker name
    """
    print(f"\n{'-'*80}")
    print(f"Running {picker_name} on {network} network data")
    print(f"{'-'*80}")
    
    if picker is None:
        print(f"⚠ Picker {picker_name} is not available, skipping...")
        return pd.DataFrame()
    
    all_picks_df = []
    total_windows = len(cached_data)
    window_count = 0
    
    # Process each cached time window
    for time_key, stream in cached_data.items():
        window_count += 1
        
        try:
            # Filter stream for specific network
            network_stream = stream.select(network=network)
            
            if len(network_stream) == 0:
                continue
            
            # Divide traces into batches to manage memory
            batch_streams = []
            current_batch = Stream()
            
            for trace in network_stream:
                current_batch += trace
                
                # When batch is full, save it and start a new one
                if len(current_batch) >= BATCH_SIZE:
                    batch_streams.append(current_batch.copy())
                    current_batch = Stream()
            
            # Add any remaining traces
            if len(current_batch) > 0:
                batch_streams.append(current_batch)
            
            # Process each batch
            for batch_idx, batch_stream in enumerate(batch_streams):
                try:
                    # Run the picker model on this batch
                    picks = picker.classify(
                        batch_stream,
                        P_threshold=P_THRESHOLD,
                        S_threshold=S_THRESHOLD
                    ).picks
                    
                    # Process and store picks
                    if picks:
                        batch_picks = []
                        for p in picks:
                            # Parse trace ID (format: network.station.location.channel)
                            trace_parts = p.trace_id.split(".")
                            batch_picks.append({
                                "id": p.trace_id,
                                "network": trace_parts[0],
                                "station": trace_parts[1],
                                "channel": trace_parts[3] if len(trace_parts) > 3 else "",
                                "timestamp": p.peak_time.datetime,
                                "prob": p.peak_value,
                                "type": p.phase.lower(),  # 'p' or 's'
                                "picker": picker_name
                            })
                        all_picks_df.extend(batch_picks)
                    
                    # Clear memory after each batch
                    del picks
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ⚠ Error processing batch {batch_idx}: {e}")
                    continue
            
            # Report progress
            if window_count % 10 == 0 or window_count == total_windows:
                pick_counter = Counter([p['type'] for p in all_picks_df if p['network'] == network])
                print(f"  Progress: {window_count}/{total_windows} windows | "
                      f"Picks so far: {pick_counter.get('p', 0)} P, {pick_counter.get('s', 0)} S")
            
        except Exception as e:
            print(f"  ⚠ Error processing window {time_key}: {e}")
            continue
    
    # Convert to DataFrame and remove duplicates
    picks = pd.DataFrame(all_picks_df)
    
    if not picks.empty:
        picks = picks.drop_duplicates()
        pick_counter = Counter(picks['type'])
        print(f"\n✓ Total unique picks for {network}: {len(picks)}")
        print(f"  P-waves: {pick_counter.get('p', 0)}")
        print(f"  S-waves: {pick_counter.get('s', 0)}")
    else:
        print(f"\n⚠ No picks found for {network} with {picker_name}")
    
    return picks


def process_picker_parallel(args):
    """
    Wrapper function to process a single picker in parallel.
    
    This function is called by the parallel processing framework.
    It initializes a picker, runs it on data, then cleans up memory.
    
    Args:
        args (tuple): (picker_name, picker_config, network, cached_data)
        
    Returns:
        tuple: (picker_name, picks_dataframe)
    """
    picker_name, picker_config, network, cached_data = args
    
    print(f"\n▶ Starting {picker_name} for network {network}")
    
    # Initialize the picker
    picker = initialize_picker(picker_config)
    
    if picker is None:
        return picker_name, pd.DataFrame()
    
    # Run picking
    picks = run_picks_on_cached_data(cached_data, network, picker, picker_name)
    
    # Clean up memory (important for parallel processing!)
    del picker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ Completed {picker_name}")
    
    return picker_name, picks


def runstations(tstart, tend, networks, channels):
    """
    Get station information and coordinates for all networks.
    
    Retrieves station metadata from FDSN and converts coordinates
    to both geographic (lat/lon) and local UTM (x/y/z in km).
    
    Args:
        tstart (str): Start date
        tend (str): End date
        networks (str): Network codes
        channels (str): Channel patterns
        
    Returns:
        pandas.DataFrame: Station information with columns:
            - id: Station ID (network.station.)
            - longitude, latitude: Geographic coordinates
            - elevation(m): Elevation in meters
            - x(km), y(km), z(km): Local UTM coordinates in kilometers
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: GETTING STATION INFORMATION")
    print(f"{'='*80}")
    
    start_date = UTCDateTime(f"{tstart}T00:00:00")
    end_date = UTCDateTime(f"{tend}T00:00:00")

    stations = []
    
    try:
        # Get station inventory from FDSN
        all_networks = f"{ZZ_NETWORK},{PR_NETWORK}"
        print(f"Querying {CLIENT_NAME} for stations...")
        
        inv = client.get_stations(
            network=all_networks, 
            station="*", 
            location="*",
            channel=channels, 
            starttime=start_date, 
            endtime=end_date
        )

        # Process inventory
        for network in inv:
            for station in network:
                # Skip excluded stations
                if station.code in EXCLUDED_STATIONS:
                    print(f"  ⊘ Excluding station {network.code}.{station.code}")
                    continue
                
                # Only process our networks of interest
                if network.code in [ZZ_NETWORK, PR_NETWORK]:
                    try:
                        stations.append({
                            "id": f"{network.code}.{station.code}.",
                            "longitude": station.longitude,
                            "latitude": station.latitude,
                            "elevation(m)": station.elevation
                        })
                    except AttributeError as e:
                        print(f"  ⚠ Error accessing attributes for {station.code}: {e}")

        # Convert to DataFrame
        stations = pd.DataFrame(stations)

        if not stations.empty:
            # Transform geographic coordinates to local UTM
            # This is required by GAMMA for distance calculations
            print("Converting coordinates to local UTM...")
            
            stations["x(km)"] = stations.apply(
                lambda row: transformer.transform(row["longitude"], row["latitude"])[0] / 1e3, 
                axis=1
            )
            stations["y(km)"] = stations.apply(
                lambda row: transformer.transform(row["longitude"], row["latitude"])[1] / 1e3, 
                axis=1
            )
            # Elevation is positive up, but depth (z) is positive down
            stations["z(km)"] = -stations["elevation(m)"] / 1e3

            print(f"\n✓ Total stations found: {len(stations)}")
            print(f"  {ZZ_NETWORK} network: {len(stations[stations['id'].str.startswith(ZZ_NETWORK)])}")
            print(f"  {PR_NETWORK} network: {len(stations[stations['id'].str.startswith(PR_NETWORK)])}")
        else:
            print("⚠ No stations found!")

    except Exception as e:
        print(f"✗ Error getting station information: {e}")
        stations = pd.DataFrame()
    
    return stations


def convert_utm_to_latlon(x_km, y_km):
    """
    Convert UTM coordinates back to latitude/longitude.
    
    Args:
        x_km (float): East-West coordinate in kilometers
        y_km (float): North-South coordinate in kilometers
        
    Returns:
        tuple: (latitude, longitude) in degrees
    """
    # Convert km to meters
    lon, lat = transform(utm_proj, wgs84_proj, x_km * 1000, y_km * 1000)
    return lat, lon


def aggregate_picks(all_picks_data, min_pickers=MIN_PICKERS_FOR_AGGREGATION, 
                   time_tolerance=TIME_TOLERANCE_SECONDS):
    """
    Aggregate picks detected by multiple pickers.
    
    This function combines picks from different models that detected the
    same seismic phase. Only picks detected by at least min_pickers models
    within time_tolerance seconds are kept.
    
    Algorithm:
        1. Group picks by station and phase type
        2. Find clusters of picks within time_tolerance
        3. Keep only clusters with >= min_pickers detections
        4. Average the time and probability
    
    Args:
        all_picks_data (dict): Dictionary of {picker_name: picks_dataframe}
        min_pickers (int): Minimum number of pickers required
        time_tolerance (float): Time window in seconds
        
    Returns:
        pandas.DataFrame: Aggregated picks with additional columns:
            - contributing_pickers: Comma-separated list of pickers
            - num_contributing_pickers: Count of pickers that detected it
    
    Example:
        If 3 pickers detect a P-wave at station ABC within 5 seconds:
        - Pick 1: 12:34:56.123 (prob 0.85)
        - Pick 2: 12:34:56.456 (prob 0.92)
        - Pick 3: 12:34:59.789 (prob 0.78)
        Result: Aggregated pick at 12:34:57.456 (average), prob 0.85 (average)
    """
    print(f"\n{'='*80}")
    print(f"STEP 4: AGGREGATING PICKS")
    print(f"{'='*80}")
    print(f"Minimum pickers required: {min_pickers}")
    print(f"Time tolerance: {time_tolerance} seconds")
    
    # Combine all picks from all pickers
    all_picks = []
    for picker_name, picks_df in all_picks_data.items():
        if not picks_df.empty:
            picks_copy = picks_df.copy()
            all_picks.append(picks_copy)
    
    if not all_picks:
        print("⚠ No picks available for aggregation")
        return pd.DataFrame()
    
    combined_picks = pd.concat(all_picks, ignore_index=True)
    print(f"\nTotal picks from all pickers: {len(combined_picks)}")
    
    # Count picks by picker
    picker_counts = Counter(combined_picks['picker'])
    print("\nPicks per picker:")
    for picker, count in sorted(picker_counts.items()):
        print(f"  {picker}: {count}")
    
    # Group picks by station and phase type
    aggregated_picks = []
    grouped = combined_picks.groupby(['station', 'type'])
    
    print(f"\nProcessing {len(grouped)} station-phase combinations...")
    
    for (station, phase_type), group in grouped:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Find clusters of picks within time tolerance
        used_indices = set()
        
        for i, pick1 in group.iterrows():
            if i in used_indices:
                continue
                
            # Find all picks within time tolerance
            cluster_picks = []
            cluster_indices = set()
            
            for j, pick2 in group.iterrows():
                if j in used_indices:
                    continue
                    
                time_diff = abs((pick1['timestamp'] - pick2['timestamp']).total_seconds())
                
                # If within tolerance, add to cluster
                if time_diff <= time_tolerance:
                    cluster_picks.append(pick2)
                    cluster_indices.add(j)
            
            # Only keep if enough pickers agreed
            if len(cluster_picks) >= min_pickers:
                # Calculate average timestamp and probability
                avg_timestamp = pd.to_datetime([p['timestamp'] for p in cluster_picks]).mean()
                avg_prob = np.mean([p['prob'] for p in cluster_picks])
                picker_names = [p['picker'] for p in cluster_picks]
                
                # Create aggregated pick
                aggregated_pick = {
                    "id": f"{pick1['network']}.{pick1['station']}.",
                    "network": pick1['network'],
                    "station": pick1['station'],
                    "channel": pick1['channel'],
                    "timestamp": avg_timestamp,
                    "prob": avg_prob,
                    "type": phase_type,
                    "picker": f"AGGREGATED({len(cluster_picks)}pickers)",
                    "contributing_pickers": ",".join(picker_names),
                    "num_contributing_pickers": len(cluster_picks)
                }
                
                aggregated_picks.append(aggregated_pick)
                used_indices.update(cluster_indices)
    
    aggregated_df = pd.DataFrame(aggregated_picks)
    
    if not aggregated_df.empty:
        print(f"\n✓ Aggregated picks created: {len(aggregated_df)}")
        
        phase_dist = Counter(aggregated_df['type'])
        print(f"  P-waves: {phase_dist.get('p', 0)}")
        print(f"  S-waves: {phase_dist.get('s', 0)}")
        
        picker_dist = Counter(aggregated_df['num_contributing_pickers'])
        print("\nPicks by number of contributing pickers:")
        for num, count in sorted(picker_dist.items()):
            print(f"  {num} pickers: {count} picks")
    else:
        print("\n⚠ No aggregated picks found")
        print("  Try: Decrease min_pickers or increase time_tolerance")
    
    return aggregated_df


def runevents(picks, stations, output_dir, picker_combination):
    """
    Run event association using GAMMA algorithm.
    
    This function takes seismic phase picks and station information,
    then uses GAMMA to associate picks into earthquake events.
    
    Args:
        picks (pandas.DataFrame): Phase picks with timing information
        stations (pandas.DataFrame): Station locations and metadata
        output_dir (str): Directory to save results
        picker_combination (str): Name for this picker combination
        
    Returns:
        tuple: (catalog_dataframe, picks_with_assignments_dataframe)
            or (empty_df, empty_df) if association fails
    """
    print(f"\n{'='*80}")
    print(f"STEP 5: ASSOCIATING EVENTS WITH GAMMA")
    print(f"{'='*80}")
    print(f"Picker combination: {picker_combination}")
    
    config = setup_gamma_config()

    if picks.empty:
        print(f"⚠ No picks available for association")
        return pd.DataFrame(), pd.DataFrame()

    if stations.empty:
        print(f"⚠ No stations available for association")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Number of picks: {len(picks)}")
    print(f"Number of stations: {len(stations)}")
    
    pick_dist = Counter(picks['type'])
    print(f"Pick distribution: {pick_dist.get('p', 0)} P, {pick_dist.get('s', 0)} S")

    try:
        # Run GAMMA association
        print("\nRunning GAMMA association...")
        catalogs, assignments = association(picks, stations, config, method=config["method"])
        catalog = pd.DataFrame(catalogs)

        if catalog.empty:
            print(f"⚠ No events found")
            print("  Possible solutions:")
            print("  - Decrease min_picks_per_eq in GAMMA_CONFIG")
            print("  - Increase dbscan_eps")
            print("  - Check if picks have correct format")
            return pd.DataFrame(), pd.DataFrame()

        # Create assignments DataFrame
        assignments_df = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
        
        # Merge picks with assignments
        full_assignments = assignments_df.merge(
            picks[["id", "network", "station", "timestamp", "prob", "type", "picker"]],
            left_on="pick_idx",
            right_index=True,
            how="left"
        )

        # Convert UTM coordinates back to lat/lon
        print("Converting event locations to geographic coordinates...")
        catalog[['latitude', 'longitude']] = catalog.apply(
            lambda row: convert_utm_to_latlon(row['x(km)'], row['y(km)']), 
            axis=1, result_type='expand'
        )

        print(f"\n✓ Found {len(catalog)} events")
        print(f"  Total assigned picks: {len(full_assignments)}")
        print(f"  Average picks per event: {len(full_assignments)/len(catalog):.1f}")
        
        # Print event statistics
        print("\nEvent statistics:")
        print(f"  Depth range: {catalog['z(km)'].min():.1f} - {catalog['z(km)'].max():.1f} km")
        print(f"  Magnitude range: {catalog['magnitude'].min():.1f} - {catalog['magnitude'].max():.1f}")

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        picks_csv = os.path.join(output_dir, f"picks_{timestamp}.csv")
        catalog_csv = os.path.join(output_dir, f"catalog_{timestamp}.csv")

        # Save picks with assignments
        print(f"\nSaving picks to: {picks_csv}")
        with open(picks_csv, 'w') as fp:
            full_assignments.to_csv(
                fp, sep="\t", index=False,
                date_format='%Y-%m-%dT%H:%M:%S.%f',
                columns=["pick_idx", "event_idx", "prob_gamma", "id", "network", "station", 
                        "timestamp", "prob", "type", "picker"]
            )

        # Save catalog
        print(f"Saving catalog to: {catalog_csv}")
        with open(catalog_csv, 'w') as fp:
            catalog.to_csv(
                fp, sep="\t", index=False,
                float_format="%.3f",
                date_format='%Y-%m-%dT%H:%M:%S.%f',
                columns=["event_index", "time", "magnitude", "sigma_time", "sigma_amp", "cov_time_amp", 
                        "gamma_score", "num_picks", "num_p_picks", "num_s_picks", 
                        "x(km)", "y(km)", "z(km)", "longitude", "latitude"]
            )

        print(f"✓ Results saved to {output_dir}")
        return catalog, full_assignments

    except Exception as e:
        print(f"✗ Error in GAMMA association: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()


def main():
    """
    Main execution function - orchestrates the entire workflow.
    
    Workflow:
        1. Download and cache waveform data
        2. Get station information
        3. Run all pickers in parallel
        4. Aggregate picks from multiple pickers
        5. Associate events using GAMMA
        6. Save results and summary
    """
    print("="*80)
    print("PICKER AGGREGATOR: Multi-Model Earthquake Detection System")
    print("="*80)
    print(f"Developed by: Asiye Aziz Zanjani, 2025")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Print configuration summary
    print("\nConfiguration:")
    print(f"  Time range: {TSTART} to {TEND}")
    print(f"  Networks: {ZZ_NETWORK} (OBS), {PR_NETWORK} (Land)")
    print(f"  OBS pickers: {len(ZZ_PICKERS)}")
    print(f"  Land pickers: {len(PR_PICKERS)}")
    print(f"  Aggregation: {MIN_PICKERS_FOR_AGGREGATION}+ pickers, {TIME_TOLERANCE_SECONDS}s tolerance")
    print(f"  Parallel workers: {PARALLEL_WORKERS}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    
    # STEP 1: Download and cache all data
    cached_data = download_and_cache_data(TSTART, TEND, f"{ZZ_NETWORK},{PR_NETWORK}", 
                                         CHANNELS, TIME_OVERLAP)
    
    if not cached_data:
        print("\n✗ No data downloaded. Exiting.")
        print("  Check: Network codes, date range, FDSN client connection")
        return
    
    # STEP 2: Get stations
    stations = runstations(TSTART, TEND, f"{ZZ_NETWORK},{PR_NETWORK}", CHANNELS)
    
    if stations.empty:
        print("\n✗ No stations found. Exiting.")
        return
    
    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # STEP 3: Process all pickers in parallel
    print(f"\n{'='*80}")
    print(f"STEP 3: PROCESSING PICKERS IN PARALLEL")
    print(f"{'='*80}")
    
    # Prepare arguments for parallel processing
    parallel_args = []
    
    # Add ZZ network pickers
    for picker_name, picker_config in ZZ_PICKERS.items():
        parallel_args.append((picker_name, picker_config, ZZ_NETWORK, cached_data))
    
    # Add PR network pickers
    for picker_name, picker_config in PR_PICKERS.items():
        parallel_args.append((picker_name, picker_config, PR_NETWORK, cached_data))
    
    print(f"Total pickers to process: {len(parallel_args)}")
    print(f"Parallel workers: {min(PARALLEL_WORKERS, len(parallel_args))}")
    
    # Process pickers in parallel
    all_picks_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, len(parallel_args))) as executor:
        future_to_picker = {executor.submit(process_picker_parallel, args): args[0] for args in parallel_args}
        
        for future in concurrent.futures.as_completed(future_to_picker):
            picker_name = future_to_picker[future]
            try:
                returned_name, picks = future.result()
                all_picks_data[returned_name] = picks
            except Exception as e:
                print(f"✗ Error processing {picker_name}: {e}")
    
    # STEP 4: Create aggregated picks
    aggregated_picks = aggregate_picks(all_picks_data, MIN_PICKERS_FOR_AGGREGATION, 
                                      TIME_TOLERANCE_SECONDS)
    
    # STEP 5: Process aggregated picks with GAMMA
    if not aggregated_picks.empty:
        aggregated_output_dir = os.path.join(BASE_OUTPUT_DIR, AGGREGATED_DIR)
        os.makedirs(aggregated_output_dir, exist_ok=True)
        
        result = runevents(aggregated_picks, stations, aggregated_output_dir, "AGGREGATED")
        
        if isinstance(result, tuple) and len(result) == 2:
            catalog, picks_with_assignments = result
            
            if not catalog.empty:
                # Save with standard filenames
                picks_csv_path = os.path.join(aggregated_output_dir, PICKS_FILENAME)
                catalog_csv_path = os.path.join(aggregated_output_dir, CATALOG_FILENAME)
                
                with open(picks_csv_path, 'w') as fp:
                    picks_with_assignments.to_csv(fp, sep="\t", index=False,
                                                 date_format='%Y-%m-%dT%H:%M:%S.%f')
                
                with open(catalog_csv_path, 'w') as fp:
                    catalog.to_csv(fp, sep="\t", index=False, float_format="%.3f",
                                 date_format='%Y-%m-%dT%H:%M:%S.%f')
    # STEP 6: Run GAMMA for selected combinations (optional)
    print("\n=== STEP 6: Running GAMMA for selected combinations ===")
    
    # Only process a subset of combinations to save time
    # Uncomment this section if you want to process all combinations
    
    # Separate picks by network
    zz_picks_data = {k: v for k, v in all_picks_data.items() if any(net in k for net in ZZ_PICKERS.keys())}
    pr_picks_data = {k: v for k, v in all_picks_data.items() if any(net in k for net in PR_PICKERS.keys())}
    
    # Process only the best performing combinations
    best_combinations = [
        ("OBSTransformer_obst2024", "PhaseNet_instance"),
        ("OBSTransformer_obst2024", "PhaseNet_stead"),
        ("OBSTransformer_obst2024", "PhaseNet_scedc"),
        ("OBSTransformer_obst2024", "PhaseNet_original"),
        ("PickBlue_phasenet", "PhaseNet_stead"),
        ("PickBlue_phasenet", "PhaseNet_instance"),
        ("PickBlue_phasenet", "PhaseNet_scedc"),
        ("PickBlue_phasenet", "PhaseNet_original")
    ]
    
    for zz_picker, pr_picker in best_combinations:
        if zz_picker in zz_picks_data and pr_picker in pr_picks_data:
            combination = f"{zz_picker}+{pr_picker}"
            print(f"\nProcessing combination: {combination}")
            
            # Combine picks from both networks
            zz_picks = zz_picks_data[zz_picker]
            pr_picks = pr_picks_data[pr_picker]
            
            if not zz_picks.empty and not pr_picks.empty:
                combined_picks = pd.concat([zz_picks, pr_picks], ignore_index=True)
            elif not zz_picks.empty:
                combined_picks = zz_picks
            elif not pr_picks.empty:
                combined_picks = pr_picks
            else:
                continue
            
            # Create output directory and run GAMMA
            combo_output_dir = os.path.join(BASE_OUTPUT_DIR, combination)
            result = runevents(combined_picks, stations, combo_output_dir, combination)
    
    # Save processing summary
    print(f"\n{'='*80}")
    print("CREATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary_file = os.path.join(BASE_OUTPUT_DIR, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PICKER AGGREGATOR - PROCESSING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Time range: {TSTART} to {TEND}\n")
        f.write(f"Networks: {ZZ_NETWORK}, {PR_NETWORK}\n")
        f.write(f"Channels: {CHANNELS}\n")
        f.write(f"Total cached data chunks: {len(cached_data)}\n")
        f.write(f"Total stations: {len(stations)}\n")
        f.write(f"Total pickers processed: {len(all_picks_data)}\n\n")
        
        f.write("Picker Results:\n")
        f.write("-"*80 + "\n")
        for picker_name, picks in all_picks_data.items():
            pick_dist = Counter(picks['type']) if not picks.empty else Counter()
            f.write(f"  {picker_name}:\n")
            f.write(f"    Total: {len(picks)} picks\n")
            f.write(f"    P-waves: {pick_dist.get('p', 0)}\n")
            f.write(f"    S-waves: {pick_dist.get('s', 0)}\n")
        
        f.write(f"\nAggregated Results:\n")
        f.write("-"*80 + "\n")
        if not aggregated_picks.empty:
            agg_dist = Counter(aggregated_picks['type'])
            f.write(f"  Total aggregated picks: {len(aggregated_picks)}\n")
            f.write(f"  P-waves: {agg_dist.get('p', 0)}\n")
            f.write(f"  S-waves: {agg_dist.get('s', 0)}\n")
            
            if 'catalog' in locals() and not catalog.empty:
                f.write(f"\n  Events detected: {len(catalog)}\n")
                f.write(f"  Average picks per event: {len(picks_with_assignments)/len(catalog):.1f}\n")
        else:
            f.write("  No aggregated picks created\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results directory: {BASE_OUTPUT_DIR}")
    print(f"Aggregated results: {os.path.join(BASE_OUTPUT_DIR, AGGREGATED_DIR)}")
    print(f"Cached data: {DATA_CACHE_DIR}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
