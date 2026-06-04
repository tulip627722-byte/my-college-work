"""Apply 2D mask to crop frame 0 point cloud, visualize result"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Load 2D mask (1920x1080 RGB resolution)
mask_raw = np.fromfile(os.path.join(OUT, "mask_2d.png"), dtype=np.uint8)
mask_rgb = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
print(f"Mask: {mask_rgb.shape}, active pixels: {np.sum(mask_rgb > 0)}")

# Load frame 0 depth
d_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
h, w = depth.shape  # 424, 512

fx = fy = 365.456
cx, cy = 254.878, 205.395

# Full point cloud
v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]

# Map depth pixels -> RGB pixels for mask lookup
u_rgb = np.clip((uv * 1920 / 512).astype(int), 0, 1919)
v_rgb = np.clip((vv * 1080 / 424).astype(int), 0, 1079)

# Mask filter
in_mask = mask_rgb[v_rgb, u_rgb] > 0
zv = zv[in_mask]; uv = uv[in_mask]; vv = vv[in_mask]
u_rgb = u_rgb[in_mask]; v_rgb = v_rgb[in_mask]

# Compute 3D
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

# Color
c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
colors = color_rgb[v_rgb, u_rgb] / 255.0

print(f"\nAfter mask: {len(pts)} points")
print(f"X: [{x.min():.4f}, {x.max():.4f}]")
print(f"Y: [{y.min():.4f}, {y.max():.4f}]")
print(f"Z: [{zv.min():.4f}, {zv.max():.4f}]")

# Depth range filter (0.4m ~ 2.0m)
depth_mask = (zv > 0.4) & (zv < 2.0)
pts = pts[depth_mask]; colors = colors[depth_mask]
print(f"After depth filter (0.4~2.0m): {len(pts)} points")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

save_path = os.path.join(OUT, "debug_frame000_manual.ply")
o3d.io.write_point_cloud(save_path, pcd)
print(f"\nSaved: debug_frame000_manual.ply ({len(pts)} points)")

# Also save XYZ ranges for later batch processing
ranges = {
    "X": [float(x.min()), float(x.max())],
    "Y": [float(y.min()), float(y.max())],
    "Z": [float(zv.min()), float(zv.max())],
}
with open(os.path.join(OUT, "crop_ranges.json"), "w") as f:
    json.dump(ranges, f, indent=2)

print(f"\nCrop ranges saved for batch processing:")
print(f"  X: [{ranges['X'][0]:.4f}, {ranges['X'][1]:.4f}]")
print(f"  Y: [{ranges['Y'][0]:.4f}, {ranges['Y'][1]:.4f}]")
print(f"  Z: [{ranges['Z'][0]:.4f}, {ranges['Z'][1]:.4f}]")
print(f"\nOpen debug_frame000_manual.ply to verify, then tell me to batch process all 36 frames.")
