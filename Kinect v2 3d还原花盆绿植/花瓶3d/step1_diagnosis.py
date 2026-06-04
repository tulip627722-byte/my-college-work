"""Step 1 — 数据诊断"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = "/e/花瓶3d"
OUT = os.path.join(DATA, "output")
os.makedirs(OUT, exist_ok=True)

# 1. 读取 depth/000.png
depth = cv2.imread(os.path.join(DATA, "depth", "000.png"), cv2.IMREAD_UNCHANGED)
print(f"[深度图] 分辨率: {depth.shape}  dtype: {depth.dtype}  min: {depth.min()}  max: {depth.max()}")
nonzero_ratio = np.count_nonzero(depth) / depth.size
print(f"[深度图] 非零像素比例: {nonzero_ratio:.4f} ({np.count_nonzero(depth)}/{depth.size})")

# 2. 读取 color/000.png
color = cv2.imread(os.path.join(DATA, "color", "000.png"), cv2.IMREAD_COLOR)
print(f"[RGB图]   分辨率: {color.shape}  dtype: {color.dtype}")

# 3. 深度图转点云
h, w = depth.shape
fx, fy = 365.456, 365.456
cx, cy = 254.878, 205.395

# 生成像素网格
v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0  # mm -> m
valid = z > 0.001  # 过滤零值
z_valid = z[valid]
u_valid = u[valid]
v_valid = v[valid]

x = (u_valid - cx) * z_valid / fx
y = (v_valid - cy) * z_valid / fy

pts = np.stack([x, y, z_valid], axis=1)
print(f"\n[帧0点云] 点数: {pts.shape[0]}")
print(f"[帧0点云] X范围: [{x.min():.4f}, {x.max():.4f}] m")
print(f"[帧0点云] Y范围: [{y.min():.4f}, {y.max():.4f}] m")
print(f"[帧0点云] Z范围: [{z_valid.min():.4f}, {z_valid.max():.4f}] m")

# 4. 保存帧0原始点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_raw.ply"), pcd)
print(f"\n[保存] debug_frame000_raw.ply 已生成，请目视检查")
print(f"[诊断完成] Step 1 完毕")
