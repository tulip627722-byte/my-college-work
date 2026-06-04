"""
Shared configuration for Kinect V2 3D reconstruction pipeline.
All parameters in one place.
"""
import os
import numpy as np

# ============================================================
# Paths
# ============================================================
DATA_DIR = r"E:\花瓶3d"
COLOR_DIR = os.path.join(DATA_DIR, "color")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
OUTPUT_DIR = os.path.join(DATA_DIR, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_FRAMES = 36

# ============================================================
# Kinect V2 Calibration (from libfreenect2 firmware defaults)
# ============================================================

# Depth/IR camera intrinsics (512 x 424)
DEPTH_FX = 368.0966
DEPTH_FY = 368.0966
DEPTH_CX = 261.6966
DEPTH_CY = 202.5222
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

# Depth camera radial distortion (no tangential)
DEPTH_K1 = 0.094
DEPTH_K2 = -0.271
DEPTH_K3 = 0.098
DEPTH_P1 = 0.0
DEPTH_P2 = 0.0

# Camera matrix and distortion vector for OpenCV
K_DEPTH = np.array([
    [DEPTH_FX, 0, DEPTH_CX],
    [0, DEPTH_FY, DEPTH_CY],
    [0, 0, 1]
], dtype=np.float64)

D_DEPTH = np.array([DEPTH_K1, DEPTH_K2, DEPTH_P1, DEPTH_P2, DEPTH_K3], dtype=np.float64)

# Color camera intrinsics (1920 x 1080)
COLOR_FX = 1081.37
COLOR_FY = 1081.37
COLOR_CX = 959.5
COLOR_CY = 539.5
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080

K_COLOR = np.array([
    [COLOR_FX, 0, COLOR_CX],
    [0, COLOR_FY, COLOR_CY],
    [0, 0, 1]
], dtype=np.float64)

# Extrinsic: depth camera → color camera
# Approximate: ~52mm baseline along X, near-identity rotation
R_DEPTH_TO_COLOR = np.eye(3, dtype=np.float64)
T_DEPTH_TO_COLOR = np.array([-0.052, 0.0, 0.0], dtype=np.float64)

# ============================================================
# Turntable parameters
# ============================================================
ROT_CX = -0.0346  # Rotation center X (world coords)
ROT_CZ = 0.7861   # Rotation center Z (world coords)
DEG_PER_FRAME = 360.0 / NUM_FRAMES  # 10 degrees

# ============================================================
# Depth filtering
# ============================================================
DEPTH_MIN_M = 0.3   # Minimum valid depth (meters)
DEPTH_MAX_M = 1.5   # Maximum valid depth (meters)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

# ============================================================
# Segmentation
# ============================================================
# Pot
POT_Y_MIN = 0.08   # meters (above ground)
POT_Y_MAX = 0.20   # meters
POT_DEPTH_MIN = 0.3
POT_DEPTH_MAX = 1.5

# Plant
PLANT_Y_MAX = 0.10  # meters (below this = plant)

# HSV thresholds for green (OpenCV: H∈[0,180], S∈[0,255], V∈[0,255])
# H=35°→17 in OpenCV half-range, H=85°→42
GREEN_H_MIN = 17    # 35° / 2
GREEN_H_MAX = 42    # 85° / 2
GREEN_S_MIN = 77    # 0.3 * 255 ≈ 77

# ============================================================
# Voxel & SOR
# ============================================================
VOXEL_SIZE = 0.003  # 3 mm

# Pot SOR (strict)
POT_SOR_NB = 16
POT_SOR_STD = 2.0

# Plant SOR (loose)
PLANT_SOR_NB = 8
PLANT_SOR_STD = 3.0

# ============================================================
# ICP
# ============================================================
# Pot ICP
POT_ICP_THRESHOLD = 0.05  # meters
POT_ICP_MAX_ITER = 50

# Plant ICP
PLANT_ICP_THRESHOLD = 0.03  # meters
PLANT_ICP_MAX_ITER = 50
PLANT_ICP_MIN_FITNESS = 0.2
PLANT_ICP_MAX_RMSE = 0.01

# ============================================================
# TSDF
# ============================================================
TSDF_VOXEL = 0.003     # 3 mm
TSDF_TRUNC = 0.015     # 15 mm
TSDF_DEPTH_MIN = 0.3   # meters
TSDF_DEPTH_MAX = 1.5   # meters

# ============================================================
# Outlier removal (plant per-frame before TSDF)
# ============================================================
PLANT_RADIUS_NB = 5
PLANT_RADIUS_R = 0.01  # 10 mm

# ============================================================
# Smoothing
# ============================================================
TAUBIN_ITERATIONS = 3

# ============================================================
# Normal estimation
# ============================================================
NORMAL_RADIUS = 0.01   # 10 mm
NORMAL_MAX_NN = 30
