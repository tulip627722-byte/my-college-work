"""Open3D interactive viewer for frame 0 - inspect and close"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"

d_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
h, w = depth.shape

fx = fy = 365.456
cx, cy = 254.878, 205.395

v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

# Color
c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
u_rgb = np.clip((uv * 1920 / 512).astype(int), 0, 1919)
v_rgb = np.clip((vv * 1080 / 424).astype(int), 0, 1079)
colors = color_rgb[v_rgb, u_rgb] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)
# Downsample for speed
pcd = pcd.uniform_down_sample(every_k_points=4)

# Add origin frame for reference
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

print(f"Points: {len(pcd.points)}")
print("Look at the vase. Note approximate ranges:")
print("  X: left-right (across the vase)")
print("  Y: up-down (height)")
print("  Z: front-back (depth from camera)")
print("\nClose window when done.")

o3d.visualization.draw_geometries(
    [pcd, origin],
    window_name="Frame 0 - Close when done inspecting",
    width=1280, height=720,
)
