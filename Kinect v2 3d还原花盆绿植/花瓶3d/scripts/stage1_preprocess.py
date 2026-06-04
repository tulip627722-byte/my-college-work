"""
Stage 1: Single-frame preprocessing with HSV+RGB dual validation
Output: pot + plant point clouds for frame 0, visual check
"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

FX, FY = 365.456, 365.456
CX, CY = 254.878, 205.395
ROT_CX, ROT_CZ = -0.0346, 0.7861

# ============================================================
# Load frame 0
# ============================================================
d_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)

c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_bgr = cv2.imdecode(c_raw, cv2.IMREAD_COLOR)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

# ============================================================
# Bilateral filter depth
# ============================================================
depth_bf = cv2.bilateralFilter(depth.astype(np.float32), 5, 30, 30)
depth_m = depth_bf / 1000.0

# ============================================================
# Generate 3D points
# ============================================================
vv, uu = np.mgrid[0:424, 0:512]
zz = depth_m
ok = zz > 0.01
zv, uv, vv = zz[ok], uu[ok], vv[ok]
x = (uv - CX) * zv / FX
y = (vv - CY) * zv / FY
pts = np.stack([x, y, zv], axis=1)

# Color lookup
ur = np.clip((uv * 1920 // 512).astype(int), 0, 1919)
vr = np.clip((vv * 1080 // 424).astype(int), 0, 1079)
clr_rgb = color_rgb[vr, ur]  # (N,3) uint8
clr_hsv = color_hsv[vr, ur]  # (N,3) uint8

# ============================================================
# Load spatial mask (dilated 35px)
# ============================================================
mask_raw = np.fromfile(os.path.join(DATA, "output", "mask_2d.png"), dtype=np.uint8)
mask_2d = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
spatial_mask = cv2.dilate(mask_2d, kernel, iterations=1)
in_mask = spatial_mask[vr, ur] > 0

# ============================================================
# Color edge constraint: fix depth bleeding at object boundaries
# ============================================================
gray_rgb = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_rgb, 50, 150)
edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8))
edges_pts = edges_dilated[vr, ur] > 0
# At edge points, check for depth jumps vs neighbors
# Simple: mark edge points where depth differs > 30mm from non-edge neighbor median
# For now: penalize edge points by flagging them
is_edge_point = edges_pts

# ============================================================
# POT segmentation: HSV low-sat + RGB g<r + spatial + Y + depth
# ============================================================
h, s, v = clr_hsv[:,0].astype(float), clr_hsv[:,1].astype(float), clr_hsv[:,2].astype(float)
r, g, b = clr_rgb[:,0].astype(float), clr_rgb[:,1].astype(float), clr_rgb[:,2].astype(float)

# Pot = low saturation (not green) + within mask + Y range
is_pot_hsv = (s < 60) & (v > 40) & (v < 230)   # low-saturation, not too dark/bright
is_pot_rgb = g < r * 1.05                        # not greener than red
is_pot = (in_mask & is_pot_hsv & is_pot_rgb &
          (pts[:,1] > 0.06) & (pts[:,1] < 0.20) &
          (pts[:,2] > 0.4) & (pts[:,2] < 2.0))
pts_pot = pts[is_pot]

pcd_pot = o3d.geometry.PointCloud()
pcd_pot.points = o3d.utility.Vector3dVector(pts_pot)
pcd_pot = pcd_pot.voxel_down_sample(0.003)
pcd_pot, _ = pcd_pot.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_pot.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
pts_pot_clean = np.asarray(pcd_pot.points)

# ============================================================
# PLANT segmentation: HSV green + RGB green + spatial + Y + depth
# ============================================================
# HSV green: H in [35, 85], S > 30, V > 30
is_plant_hsv = (h >= 35) & (h <= 85) & (s > 30) & (v > 30)
# RGB green: g > r*0.95
is_plant_rgb = g > r * 0.95
# Combined
is_plant = (in_mask & is_plant_hsv & is_plant_rgb &
            (pts[:,1] < 0.10) & (pts[:,1] > -0.15) &
            (pts[:,2] > 0.3) & (pts[:,2] < 2.5) &
            ~is_edge_point)  # remove depth bleeding at edges
pts_plant = pts[is_plant]

pcd_plant = o3d.geometry.PointCloud()
pcd_plant.points = o3d.utility.Vector3dVector(pts_plant)
pcd_plant = pcd_plant.voxel_down_sample(0.003)
pcd_plant, _ = pcd_plant.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.0)
pcd_plant.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
pts_plant_clean = np.asarray(pcd_plant.points)

print(f"Pot:   {len(pts_pot_clean)} pts")
print(f"Plant: {len(pts_plant_clean)} pts")
print(f"Total: {len(pts_pot_clean)+len(pts_plant_clean)} pts")

# ============================================================
# Visualize: Blue=Pot, Green=Plant
# ============================================================
pcd_viz = o3d.geometry.PointCloud()
pcd_viz.points = o3d.utility.Vector3dVector(
    np.vstack([pts_pot_clean, pts_plant_clean]))
clr_viz = np.zeros((len(pts_pot_clean)+len(pts_plant_clean), 3))
clr_viz[:len(pts_pot_clean)] = [0.3, 0.4, 0.9]    # Blue = Pot
clr_viz[len(pts_pot_clean):] = [0.2, 0.8, 0.3]    # Green = Plant
pcd_viz.colors = o3d.utility.Vector3dVector(clr_viz)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

print("\nBLUE=Pot  GREEN=Plant  |  Close window to continue")
o3d.visualization.draw_geometries(
    [pcd_viz, axis],
    window_name="Stage 1 — Pot(Blue) + Plant(Green)  |  Close to confirm",
    width=1280, height=720)

# ============================================================
# Save intermediate
# ============================================================
o3d.io.write_point_cloud(os.path.join(OUT, "stage1_pot.ply"), pcd_pot)
o3d.io.write_point_cloud(os.path.join(OUT, "stage1_plant.ply"), pcd_plant)
o3d.io.write_point_cloud(os.path.join(OUT, "stage1_viz.ply"), pcd_viz)
print("\nStage 1 done. Files: stage1_pot.ply, stage1_plant.ply, stage1_viz.ply")
