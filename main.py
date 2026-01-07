import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from step1_clean_traj import run_cleaning_pipeline
from step2_enrich_net import run_network_enrichment_pipeline
from step3_map_match import run_map_match_pipeline
from step4_train_eval import run_training_pipeline

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS = {
    'raw_gps': os.path.join(ROOT_DIR, 'data', 'raw_gps'),
    'net_raw': os.path.join(ROOT_DIR, 'data', 'network_road'),
    'traj_clean': os.path.join(ROOT_DIR, 'data', 'preprocessed', 'cleaned_trajectories.csv'),
    'net_enriched': os.path.join(ROOT_DIR, 'data', 'network_road'),
    'match_out': os.path.join(ROOT_DIR, 'outputs', 'mapmatch_output')
}

def main():
    print("========================================")
    print("=== DAGRN Project Pipeline Execution ===")
    print("========================================")
    
    # --- Step 1: Trajectory Cleaning ---
    print("\n>>> Step 1: Cleaning Trajectories...")
    if os.path.exists(DIRS['raw_gps']):
         run_cleaning_pipeline(DIRS['raw_gps'], DIRS['traj_clean'])
    else:
        print(f"Warning: Raw GPS folder not found at {DIRS['raw_gps']}, skipping Step 1.")

    # --- Step 2: Network Enrichment ---
    print("\n>>> Step 2: Network Enrichment...")
    link_shp = os.path.join(DIRS['net_raw'], 'network_link_aligned.shp')
    node_shp = os.path.join(DIRS['net_raw'], 'network_nodes.shp')
    
    if os.path.exists(link_shp) and os.path.exists(node_shp):
        enr_link, enr_node = run_network_enrichment_pipeline(
            link_shp, node_shp, DIRS['net_enriched'], 
            traj_data_path=DIRS['traj_clean'] if os.path.exists(DIRS['traj_clean']) else None
        )
    else:
        print("Warning: Network shapefiles not found, skipping Step 2.")
        enr_link = link_shp

    # --- Step 3: Map Matching ---
    print("\n>>> Step 3: Map Matching...")
    target_link = os.path.join(DIRS['net_enriched'], 'network_link_enriched.shp')
    target_node = os.path.join(DIRS['net_enriched'], 'network_nodes_enriched.shp')
    
    if not os.path.exists(target_link):
        target_link = link_shp
        target_node = node_shp
    
    if os.path.exists(target_link) and os.path.exists(DIRS['traj_clean']):
         run_map_match_pipeline(target_link, target_node, DIRS['traj_clean'], DIRS['match_out'])
    else:
        print("Skipping Step 3 (Missing network or trajectory file).")

    print("\n>>> Step 4: Training & Evaluation...")
    run_training_pipeline(ROOT_DIR)
    print("\n========================================")
    print("=== Pipeline Finished Successfully ===")
    print("========================================")

if __name__ == "__main__":
    main()