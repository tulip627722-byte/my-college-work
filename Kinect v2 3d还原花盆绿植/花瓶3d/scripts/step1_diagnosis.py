"""Step 1 — Data Diagnosis"""
import numpy as np
import cv2
import open3d as o3d
import os, sys

# Hard-code the Windows path - Python on Windows handles this fine
DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")
os.makedirs(OUT, exist_ok=True)

# Read depth - use imdecode to handle Chinese paths
depth_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(depth_raw, cv2.IMREAD_UNCHANGED)
if depth is None:
    print("ERROR: Cannot read depth image"); sys.exit(1)
print(f"[Depth] shape={depth.shape} dtype={depth.dtype} min={depth.min()} max={depth.max()}")
nz = np.count_nonzero(depth)
print(f"[Depth] nonzero pixels: {nz}/{depth.size} = {nz/depth.size:.4f}")

# Read color
color_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color = cv2.imdecode(color_raw, cv2.IMREAD_COLOR)
print(f"[Color] shape={color.shape} dtype={color.dtype}")

# Depth -> point cloud
h, w = depth.shape
fx = fy = 365.456
cx, cy = 254.878, 205.395

v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0  # mm -> m
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

print(f"\n[Frame 0 Point Cloud]")
print(f"  Points: {pts.shape[0]}")
print(f"  X range: [{x.min():.4f}, {x.max():.4f}] m")
print(f"  Y range: [{y.min():.4f}, {y.max():.4f}] m")
print(f"  Z range: [{zv.min():.4f}, {zv.max():.4f}] m")

# Save raw point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
save_path = os.path.join(OUT, "debug_frame000_raw.ply")
o3d.io.write_point_cloud(save_path, pcd)
print(f"\n[Saved] {save_path}")
print("[Step 1 Complete]")
