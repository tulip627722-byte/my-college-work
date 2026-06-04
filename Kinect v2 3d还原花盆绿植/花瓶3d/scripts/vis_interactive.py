"""Interactive visualization of frame 0 raw point cloud"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"

# Read depth
depth_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(depth_raw, cv2.IMREAD_UNCHANGED)

h, w = depth.shape
fx = fy = 365.456
cx, cy = 254.878, 205.395

# Full resolution point cloud
v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

# Read color & map
color_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_bgr = cv2.imdecode(color_raw, cv2.IMREAD_COLOR)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

# Map depth pixels to RGB pixels
u_rgb = (uv * 1920 / 512).astype(int)
v_rgb = (vv * 1080 / 424).astype(int)
u_rgb = np.clip(u_rgb, 0, 1919)
v_rgb = np.clip(v_rgb, 0, 1079)
colors = color_rgb[v_rgb, u_rgb] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Downsample for speed
pcd = pcd.uniform_down_sample(every_k_points=8)

print(f"Points (downsampled every 8): {len(pcd.points)}")
print(f"\nCONTROLS:")
print(f"  Left-drag  = Rotate")
print(f"  Scroll     = Zoom")
print(f"  Right-drag = Pan")
print(f"  R          = Reset view")
print(f"  Close window to exit")
print(f"\nOpening 3D viewer...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Frame 0 Raw Point Cloud - INSPECT THEN CLOSE",
    width=1280, height=720,
)
print("Viewer closed.")
