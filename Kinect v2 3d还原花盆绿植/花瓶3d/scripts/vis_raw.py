"""Visualize frame 0 raw point cloud"""
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

# Generate point cloud
v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

# Read color
color_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_bgr = cv2.imdecode(color_raw, cv2.IMREAD_COLOR)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

# Map colors
colors = color_rgb[vv, uv] / 255.0  # normalize to [0,1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"Points: {len(pts)}")
print("Controls: Left-drag=rotate, Scroll=zoom, Right-drag=pan, Ctrl+click=select")
print("Close window to exit.")

o3d.visualization.draw_geometries_with_editing([pcd]) if False else \
o3d.visualization.draw_geometries([pcd],
    window_name="Frame 0 Raw Point Cloud - Check then close window",
    width=1280, height=720)
