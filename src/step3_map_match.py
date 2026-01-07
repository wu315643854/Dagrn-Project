# ==============================================================================
# Script 3: Map Matching & Quality Control (Path Modified Only)
# ==============================================================================

import os
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from gotrackit.map.Net import Net
from gotrackit.MapMatch import MapMatch

MATCH_PARAMS = {
    'gps_buffer': 30.0,
    'top_k': 5,
    'gps_radius': 10.0,
    'use_heading': True,
    'use_sub_net': True
}

QUALITY_THRESHOLDS = {
    'dist_ratio_min': 0.7,
    'dist_ratio_max': 1.6,
    'min_matched_edges': 5
}

warnings.filterwarnings('ignore')

def remap_network_ids(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame):
    """
    Remaps huge OSM IDs or string IDs to continuous integers to avoid OverflowError.
    Strictly following original logic.
    """
    print("[Preprocessing] Remapping Network IDs to safe integers...")
    unique_nodes = node_gdf['node_id'].unique()
    node_map = {old_id: i+1 for i, old_id in enumerate(unique_nodes)}
    node_gdf['original_node_id'] = node_gdf['node_id']
    node_gdf['node_id'] = node_gdf['node_id'].map(node_map).astype(int)
    
    if pd.api.types.is_float_dtype(link_gdf['from_node']):
        link_gdf['from_node'] = link_gdf['from_node'].astype(np.int64)
    if pd.api.types.is_float_dtype(link_gdf['to_node']):
        link_gdf['to_node'] = link_gdf['to_node'].astype(np.int64)
        
    valid_nodes = set(node_map.keys())
    link_gdf['from_node'] = link_gdf['from_node'].astype(type(unique_nodes[0]))
    link_gdf['to_node'] = link_gdf['to_node'].astype(type(unique_nodes[0]))
    
    valid_links_mask = link_gdf['from_node'].isin(valid_nodes) & link_gdf['to_node'].isin(valid_nodes)
    if (~valid_links_mask).any():
        print(f"  - Warning: Dropping { (~valid_links_mask).sum() } links with disconnected nodes.")
        link_gdf = link_gdf[valid_links_mask].copy()
    
    link_gdf['from_node'] = link_gdf['from_node'].map(node_map).astype(int)
    link_gdf['to_node'] = link_gdf['to_node'].map(node_map).astype(int)
    link_gdf['original_link_id'] = link_gdf['link_id']
    link_gdf['link_id'] = range(1, len(link_gdf) + 1)
    link_gdf['link_id'] = link_gdf['link_id'].astype(int)
    if 'dir' in link_gdf.columns:
        link_gdf['dir'] = link_gdf['dir'].fillna(0).astype(int)
    else:
        link_gdf['dir'] = 0
        
    print(f"  - Remapped {len(node_gdf)} nodes and {len(link_gdf)} links.")
    return link_gdf, node_gdf

def execute_matching(net: Net, traj_df: pd.DataFrame) -> pd.DataFrame:
    """Executes HMM-based map matching using original loop logic."""
    mpm = MapMatch(net=net, use_sub_net=MATCH_PARAMS['use_sub_net'], 
                   gps_buffer=MATCH_PARAMS['gps_buffer'], top_k=MATCH_PARAMS['top_k'], 
                   gps_radius=MATCH_PARAMS['gps_radius'], use_heading_inf=MATCH_PARAMS['use_heading'])
    
    results = []
    for tid, grp in tqdm(traj_df.groupby('trajectory_id'), desc="Matching Trajectories"):
        if len(grp) < 2: continue
        try:
            res, _, _ = mpm.execute(gps_df=grp)
            if res is not None and not res.empty:
                res['trajectory_id'] = tid
                if 'time' in grp.columns and 'time' not in res.columns:
                    grp_sorted = grp.sort_values('time')
                    limit = min(len(res), len(grp_sorted))
                    res.loc[:limit-1, 'time'] = grp_sorted['time'].values[:limit]
                results.append(res)
        except Exception:
            continue
            
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def filter_quality(matched_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filters matches based on distance ratios and sequence length."""
    if matched_df.empty:
        print("Warning: Matched DataFrame is empty. Skipping filtering.")
        return matched_df
        
    print("[QC] Evaluating match quality...")
    matched_dist = matched_df.groupby('trajectory_id')['dis_to_next'].sum()
    gps_dist = raw_df.groupby('trajectory_id')['distance_m'].sum()
    
    qc_df = pd.DataFrame({'match_d': matched_dist, 'gps_d': gps_dist})
    qc_df['ratio'] = qc_df['match_d'] / qc_df['gps_d'].replace(0, np.nan)
    seq_len = matched_df.groupby('trajectory_id').size()
    valid_ratio = (qc_df['ratio'] >= QUALITY_THRESHOLDS['dist_ratio_min']) & \
                  (qc_df['ratio'] <= QUALITY_THRESHOLDS['dist_ratio_max'])
    valid_len = seq_len >= QUALITY_THRESHOLDS['min_matched_edges']
    
    valid_ids = qc_df[valid_ratio].index.intersection(seq_len[valid_len].index)
    
    print(f"  - Total Matched: {len(qc_df)}")
    print(f"  - High Quality:  {len(valid_ids)} ({(len(valid_ids)/len(qc_df)):.1%})")
    
    return matched_df[matched_df['trajectory_id'].isin(valid_ids)]

def run_map_match_pipeline(link_path, node_path, traj_path, out_dir):
    print("[Step 3] Loading Network...")
    
    if not os.path.exists(link_path) or not os.path.exists(node_path):
        print(f"Error: Network files not found: {link_path}")
        return

    link_gdf = gpd.read_file(link_path)
    node_gdf = gpd.read_file(node_path)
    
    link_gdf, node_gdf = remap_network_ids(link_gdf, node_gdf)
    
    net = Net(link_gdf=link_gdf, node_gdf=node_gdf)
    net.init_net()
    
    raw_traj = pd.read_csv(traj_path)
    if 'time' in raw_traj.columns:
        raw_traj['time'] = pd.to_datetime(raw_traj['time'])
    
    matched_df = execute_matching(net, raw_traj)
    
    os.makedirs(out_dir, exist_ok=True)
    file_matched = os.path.join(out_dir, 'all_matched_results.csv')
    file_filtered = os.path.join(out_dir, 'all_matched_results_filtered.csv')

    if matched_df.empty:
        print("CRITICAL: No trajectories matched. Check input data coordinates.")
    else:
        matched_df.to_csv(file_matched, index=False)
        final_df = filter_quality(matched_df, raw_traj)
        final_df.to_csv(file_filtered, index=False)
        print(f"[Done] Final dataset ready for training: {file_filtered}")