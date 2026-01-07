import os
import random
import numpy as np
import torch
import logging
import sys

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[System] Global random seed set to: {seed}")

def setup_logger(log_file_path):
    """
    Sets up a logger that outputs to both console and file.
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def check_dir_exists(path, create=False):
    """
    Checks if a directory exists. Optionally creates it.
    """
    if not os.path.exists(path):
        if create:
            os.makedirs(path)
            print(f"[System] Created directory: {path}")
            return True
        else:
            print(f"[System] Warning: Directory not found: {path}")
            return False
    return True

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula for distance calculation (km).
    Useful for quick checks outside of main scripts.
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))