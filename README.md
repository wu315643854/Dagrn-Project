# DAGRN: Dual-Attentional Gated Residual Framework for Robust Travel Time Prediction

This repository contains the official implementation of the paper:  
**"A Dual-Attentional Gated Residual Framework for Robust Travel Time Prediction"** (ISPRS Int. J. Geo-Inf. 2025).

DAGRN is designed to solve the **"Cold-Start"** problem in Intelligent Transportation Systems (ITS) by effectively learning from sparse trajectory data (e.g., 2k trajectories) using **Graph-Informed Sequence Learning**.

---

## 📂 Project Structure

The project is organized as a modular Python package:

```text
DAGRN_Project/
│
├── data/                        # [Data Directory] (User needs to populate this)
│   ├── raw_gps/                 # Place raw GPS CSV files here (agent_id, lng, lat, time)
│   ├── network_road/            # Place Road Network Shapefiles (.shp, .shx, .dbf) here
│   ├── weather_data/            # Place weather CSV here (Optional)
│   └── preprocessed/            # Auto-generated intermediate files
│
├── outputs/                     # [Output Directory]
│   ├── mapmatch_output/         # Results from Map Matching (HMM)
│   └── figures/                 # Evaluation plots (Training Curves, Ablation, Sensitivity)
│
├── src/                         # [Source Code Package]
│   ├── __init__.py              # Package initialization
│   ├── step1_clean_traj.py      # Module: Trajectory cleaning, segmentation & filtering
│   ├── step2_enrich_net.py      # Module: Road attribute enrichment (Line Graph Prep)
│   ├── step3_map_match.py       # Module: HMM-based Map Matching (GoTrackIt)
│   ├── step4_train_eval.py      # Module: DAGRN Model Definition, Training & Benchmarking
│   └── utils.py                 # Module: Common utilities (Logger, Seeding, etc.)
│
├── main.py                      # [Entry Point] Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation

🛠️ Environment Setup

The code is tested on Windows 10/11 and Linux (Ubuntu 20.04) with Python 3.8+.
1. Create Environment
Recommended to use Conda to manage dependencies:

conda create -n dagrn python=3.9
conda activate dagrn

2. Install Dependencies

pip install -r requirements.txt


🚀 How to Run

1. Data Preparation
Before running the code, ensure your data is placed in the data/ folder:

Raw Trajectories: Put .csv files in data/raw_gps/. Format: agent_id, lng, lat, time.
Road Network: Put standard Shapefiles in data/network_road/.
Weather: Put wuxi_weather_2020.csv in data/weather_data/.

2. One-Click Execution
Run the entire pipeline (Cleaning → Enrichment → Matching → Training) using the main script:

python main.py

3. Pipeline Modules Description
Step 1 (Cleaning):
src/step1_clean_traj.py
Loads raw GPS points.
Segments data into trajectories based on time gaps (30min) or stay points.
Filters kinematic anomalies (drift, impossible speeds).

Step 2 (Network Enrichment):
src/step2_enrich_net.py
Standardizes Shapefiles to EPSG:4326.
Intelligently fills missing maxspeed based on road type (highway tag).
Calculates link length and free-flow travel time.

Step 3 (Map Matching):
src/step3_map_match.py
Uses HMM (Hidden Markov Model) via gotrackit to project noisy GPS points onto the road network.
Generates the topological sequence of link IDs for each trip.

Step 4 (Modeling):
src/step4_train_eval.py
Constructs the Line Graph (Dual Graph).
Trains the DAGRN model.
Performs Ablation Studies (w/o GCN, w/o FiLM, etc.).
Performs Sensitivity Analysis (Batch Size, Hidden Dim, etc.).
Generates publication-quality plots in outputs/figures/.

📊 Results & Visualization

The framework generates several key visualizations automatically:
Benchmark_Comparison_Sorted.png: Performance comparison against baselines (LSTM, Graph WaveNet, etc.).
Ablation_Study_Textured.png: Impact of Line Graph, FiLM, and Residual Fusion.
Sensitivity_Heatmap_Origin.png: Hyperparameter robustness analysis.
Training_Curves_Dual.png: Loss convergence curves.