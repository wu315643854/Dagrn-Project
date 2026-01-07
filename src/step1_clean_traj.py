# ==============================================================================
# Script 1: Raw Trajectory Data Preprocessing & Cleaning (Path Modified Only)
# ==============================================================================
import os
import glob
import warnings
import pandas as pd
import numpy as np
import skmob
from skmob.preprocessing.detection import stay_locations
from tqdm.auto import tqdm

# --- STRICTLY ORIGINAL PARAMETERS ---
PARAMS = {
    'stay_point_radius_km': 0.2,
    'stay_point_min_minutes': 10,
    'min_points': 5,
    'min_distance_m': 500,
    'min_duration_s': 120,
    'max_duration_s': 3 * 3600,
    'min_speed_kmh': 5.0,
    'max_speed_kmh': 120.0
}

warnings.filterwarnings('ignore')

def haversine_vectorized(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Calculates great-circle distance between two points (Vectorized)."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def load_and_clean_raw_data(input_path: str) -> pd.DataFrame:
    """Loads raw CSVs and performs point-level cleaning."""
    print(f"[Step 1] Loading raw data from {input_path}...")
    files = glob.glob(os.path.join(input_path, "*.csv"))
    if not files:
        if os.path.isfile(input_path):
            files = [input_path]
        else:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
            
    df_list = [pd.read_csv(f, usecols=['agent_id', 'lng', 'lat', 'time']) for f in tqdm(files, desc="Reading Files")]
    df = pd.concat(df_list, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time', 'lat', 'lng', 'agent_id'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['agent_id', 'time'], inplace=True)
    return df.reset_index(drop=True)

def segment_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Segments continuous GPS streams into trips using stay-point detection."""
    print("[Step 2] Segmenting trajectories...")
    tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lng', datetime='time', user_id='agent_id')
    stops = stay_locations(tdf, spatial_radius_km=PARAMS['stay_point_radius_km'], 
                           minutes_for_a_stop=PARAMS['stay_point_min_minutes'])
    if stops.empty:
        print("  - No stay points detected. Using raw IDs.")
        df['trajectory_id'] = df['agent_id'].astype(str) + "_0"
        return df
        
    df['trajectory_id'] = df['agent_id'].astype(str)
    return df 

def apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applies trajectory-level kinematic filters."""
    print("[Step 3] Applying strict kinematic filters...")
    df['prev_lat'] = df.groupby('trajectory_id')['lat'].shift(1)
    df['prev_lng'] = df.groupby('trajectory_id')['lng'].shift(1)
    df['dist_m'] = haversine_vectorized(df['prev_lat'], df['prev_lng'], df['lat'], df['lng']).fillna(0)
    stats = df.groupby('trajectory_id').agg(
        start_time=('time', 'min'),
        end_time=('time', 'max'),
        total_dist=('dist_m', 'sum'),
        count=('lat', 'size')
    )
    stats['duration'] = (stats['end_time'] - stats['start_time']).dt.total_seconds()
    stats['avg_speed_kmh'] = (stats['total_dist'] / stats['duration'].replace(0, 1)) * 3.6
    valid_mask = (
        (stats['count'] >= PARAMS['min_points']) &
        (stats['duration'] >= PARAMS['min_duration_s']) &
        (stats['duration'] <= PARAMS['max_duration_s']) &
        (stats['total_dist'] >= PARAMS['min_distance_m']) &
        (stats['avg_speed_kmh'] >= PARAMS['min_speed_kmh']) &
        (stats['avg_speed_kmh'] <= PARAMS['max_speed_kmh'])
    )
    
    valid_ids = stats[valid_mask].index
    print(f"  - Original Trajectories: {len(stats)}")
    print(f"  - Valid Trajectories:    {len(valid_ids)} ({(len(valid_ids)/len(stats)):.1%})")
    return df[df['trajectory_id'].isin(valid_ids)].drop(columns=['prev_lat', 'prev_lng', 'dist_m'])

def run_cleaning_pipeline(input_folder, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    raw_df = load_and_clean_raw_data(input_folder)
    segmented_df = segment_trajectories(raw_df)
    final_df = apply_strict_filters(segmented_df)
    
    final_df.to_csv(output_file, index=False)
    print(f"[Done] Cleaned dataset saved to: {output_file}")