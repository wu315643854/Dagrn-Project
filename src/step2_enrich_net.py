# ==============================================================================
# Script 2: Road Network Standardization & Enrichment
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Derive bounding box from trajectory data
# ------------------------------------------------------------------------------
def get_bounds_from_data(data_path: str, buffer: float = 0.01) -> list:
    """Reads trajectory data to determine the spatial extent for map visualization."""
    print(f"--- [Step 1] Reading trajectory bounds from {data_path} ---")
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, usecols=['lng', 'lat'])
        else:
            df = pd.read_parquet(data_path, columns=['lng', 'lat'])
    except Exception as e:
        print(f"Error reading data file: {e}")
        return None

    if df.empty:
        print("Warning: Empty dataset. Cannot determine bounds.")
        return None

    west, east = df['lng'].min(), df['lng'].max()
    south, north = df['lat'].min(), df['lat'].max()
    bounds = [west - buffer, south - buffer, east + buffer, north + buffer]
    print(f"Calculated visualization bounds: {bounds}")
    return bounds

# ------------------------------------------------------------------------------
# 2. Utility: Clean and standardize maxspeed attribute
# ------------------------------------------------------------------------------
def clean_maxspeed(speed) -> float:
    if isinstance(speed, (int, float)):
        return float(speed)
        
    if isinstance(speed, str):
        speed = speed.strip()
        if '[' in speed and ']' in speed:
            speed = speed.strip("[]").split(',')[0].strip("'\"")
        if 'mph' in speed:
            try:
                return float(speed.replace('mph', '').strip()) * 1.60934
            except:
                return np.nan
        if speed.isdigit():
            return float(speed)
            
    return np.nan

# ------------------------------------------------------------------------------
# 3. Core Logic: Enhance Existing Network Attributes
# ------------------------------------------------------------------------------
def enhance_existing_network(link_path: str, node_path: str, output_folder: str):
    """
    Loads existing shapefiles, cleans attributes, applies domain-knowledge based 
    speed limits, calculates travel time, and overwrites the files.
    """
    print(f"\n--- [Step 2] Enhancing network attributes for: {link_path} ---")
    
    if not os.path.exists(link_path) or not os.path.exists(node_path):
        print(f"Error: Input shapefiles not found at {link_path}")
        return None, None
    try:
        edges_gdf = gpd.read_file(link_path, encoding='utf-8')
        nodes_gdf = gpd.read_file(node_path, encoding='utf-8')
    except:
        edges_gdf = gpd.read_file(link_path)
        nodes_gdf = gpd.read_file(node_path)

    print(f"Loaded {len(edges_gdf)} edges and {len(nodes_gdf)} nodes.")
    if edges_gdf.crs != "EPSG:4326":
        edges_gdf = edges_gdf.to_crs("EPSG:4326")
    if nodes_gdf.crs != "EPSG:4326":
        nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
        
    if 'link_id' not in edges_gdf.columns:
        if all(col in edges_gdf.columns for col in ['u', 'v', 'key']):
            edges_gdf['link_id'] = edges_gdf.apply(lambda row: f"{row['u']}_{row['v']}_{row['key']}", axis=1)
        else:
            edges_gdf['link_id'] = edges_gdf.index.astype(str)
            
    if 'highway' in edges_gdf.columns:
        edges_gdf['highway'] = edges_gdf['highway'].astype(str).apply(
            lambda x: x.strip("[]").split(',')[0].strip("'\"")
        )
    else:
        edges_gdf['highway'] = 'unknown'
        
    print("Applying intelligent speed filling logic...")
    default_speeds = {
        'motorway': 100, 'motorway_link': 60,
        'trunk': 80, 'trunk_link': 50,
        'primary': 60, 'primary_link': 40,
        'secondary': 50, 'secondary_link': 30,
        'tertiary': 40, 'tertiary_link': 30,
        'residential': 30, 'living_street': 20,
        'unclassified': 30, 'unknown': 30
    }
    
    if 'maxspeed' in edges_gdf.columns:
        edges_gdf['speed_kph'] = edges_gdf['maxspeed'].apply(clean_maxspeed)
    else:
        edges_gdf['speed_kph'] = np.nan
        
    mask_nan = edges_gdf['speed_kph'].isna() | (edges_gdf['speed_kph'] == 0)
    edges_gdf.loc[mask_nan, 'speed_kph'] = edges_gdf.loc[mask_nan, 'highway'].map(default_speeds).fillna(30)
    
    if 'lanes' in edges_gdf.columns:
        edges_gdf['lanes_cleaned'] = edges_gdf['lanes'].astype(str).apply(
            lambda x: x.strip("[]").split(',')[0].strip("'\"")
        )
        edges_gdf['lanes_cleaned'] = pd.to_numeric(edges_gdf['lanes_cleaned'], errors='coerce').fillna(1)
    else:
        edges_gdf['lanes_cleaned'] = 1
        
    if 'oneway' in edges_gdf.columns:
        edges_gdf['dir'] = edges_gdf['oneway'].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype(int)
    else:
        edges_gdf['dir'] = 0
        
    if 'length' not in edges_gdf.columns:
        edges_gdf['length'] = edges_gdf.to_crs("EPSG:3857").geometry.length
        
    edges_gdf['travel_time_s'] = edges_gdf['length'] / (edges_gdf['speed_kph'] / 3.6)
    
    final_cols = ['link_id', 'highway', 'speed_kph', 'lanes_cleaned', 'length', 'travel_time_s', 'dir', 'geometry']
    for c in ['u', 'v', 'from_node', 'to_node']:
        if c in edges_gdf.columns: final_cols.append(c)
            
    edges_final = edges_gdf[[c for c in final_cols if c in edges_gdf.columns]].rename(columns={'lanes_cleaned': 'lanes'})

    print("--- [Step 3] Saving enhanced network to shapefiles ---")
    os.makedirs(output_folder, exist_ok=True)
    out_edge_path = os.path.join(output_folder, 'network_link_enriched.shp')
    out_node_path = os.path.join(output_folder, 'network_nodes_enriched.shp')
    
    edges_final.to_file(out_edge_path, driver='ESRI Shapefile', encoding='utf-8')
    nodes_gdf.to_file(out_node_path, driver='ESRI Shapefile', encoding='utf-8')
    
    print(f"Network attributes enriched and saved to:\n  - {out_edge_path}\n  - {out_node_path}")
    print(f"Sample:\n{edges_final[['highway', 'speed_kph', 'travel_time_s']].head(3)}")
    
    return out_edge_path, out_node_path

# ------------------------------------------------------------------------------
# 4. Interactive visualization for sub-region
# ------------------------------------------------------------------------------
def create_interactive_network_map_subregion(link_shapefile_path, output_html_path, display_bounds):
    """Generates an HTML interactive map for validation."""
    print("\n--- [Step 4] Creating interactive visualization ---")
    try:
        edges_gdf = gpd.read_file(link_shapefile_path).to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error reading network file: {e}")
        return

    west, south, east, north = display_bounds
    edges_gdf_cropped = edges_gdf.cx[west:east, south:north]
    
    if edges_gdf_cropped.empty:
        print("Warning: No network data found within specified bounds for visualization.")
        if len(edges_gdf) < 5000: edges_gdf_cropped = edges_gdf
        else: return
            
    color_map = {'motorway': 'blue', 'trunk': 'purple', 'primary': 'red', 'secondary': 'orange', 
                 'tertiary': 'yellow', 'residential': 'green', 'unclassified': 'darkgray'}
    
    def style_function(feature):
        return {'color': color_map.get(feature['properties']['highway'], 'gray'), 'weight': 3, 'opacity': 0.7}

    map_center = [edges_gdf_cropped.unary_union.centroid.y, edges_gdf_cropped.unary_union.centroid.x]
    m = folium.Map(location=map_center, tiles='CartoDB positron', zoom_start=13)

    folium.GeoJson(
        edges_gdf_cropped,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['link_id', 'highway', 'speed_kph', 'length', 'lanes'],
            aliases=['Link ID:', 'Road Class:', 'Speed (km/h):', 'Length (m):', 'Lanes:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)
    
    legend_html = '''<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px; background-color: white; padding: 10px; border: 2px solid grey;"><b>Legend</b><br>'''
    for k, v in color_map.items(): legend_html += f'<span style="color:{v}">&#9472; {k}</span><br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    if not edges_gdf_cropped.empty:
        b = edges_gdf_cropped.total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

    m.save(output_html_path)
    print(f"Interactive map saved to: {output_html_path}")
    return m

def run_network_enrichment_pipeline(link_file_path, node_file_path, output_folder, traj_data_path=None):
    if os.path.exists(link_file_path):
        out_link, out_node = enhance_existing_network(link_file_path, node_file_path, output_folder)
    else:
        print(f"CRITICAL: Shapefiles not found at {link_file_path}. Cannot proceed.")
        return None, None
    if traj_data_path and os.path.exists(traj_data_path):
        bounds = get_bounds_from_data(traj_data_path)
        
        if bounds and out_link:
            output_map_file = os.path.join(output_folder, 'interactive_road_network_map.html')
            viz_bounds = [bounds[0], bounds[1], bounds[2], bounds[3]]
            
            create_interactive_network_map_subregion(
                link_shapefile_path=out_link,
                output_html_path=output_map_file,
                display_bounds=viz_bounds
            )
    return out_link, out_node