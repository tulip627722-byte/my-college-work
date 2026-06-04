"""Generate colored PLY for all 36 frames"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output", "all_frames")
os.makedirs(OUT, exist_ok=True)

fx = fy = 365.456
cx, cy = 254.878, 205.395

for fidx in range(36):
    # Read depth
    d_raw = np.fromfile(os.path.join(DATA, "depth", f"{fidx:03d}.png"), dtype=np.uint8)
    depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
    h, w = depth.shape

    # Read color
    c_raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    color_bgr = cv2.imdecode(c_raw, cv2.IMREAD_COLOR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Depth -> points
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0.001
    zv = z[valid]; uv = u[valid]; vv = v[valid]
    x = (uv - cx) * zv / fx
    y = (vv - cy) * zv / fy
    pts = np.stack([x, y, zv], axis=1)

    # Map colors
    u_rgb = np.clip((uv * 1920 / 512).astype(int), 0, 1919)
    v_rgb = np.clip((vv * 1080 / 424).astype(int), 0, 1079)
    colors = color_rgb[v_rgb, u_rgb] / 255.0

    # Uniform downsample for manageable file size (every 2 pixels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.uniform_down_sample(every_k_points=2)

    save_path = os.path.join(OUT, f"frame_{fidx:03d}.ply")
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Frame {fidx:03d}: {len(pcd.points)} points -> {save_path}")

print(f"\nDone. {36} PLY files saved to: {OUT}")
print("Pick one frame, crop manually, save cropped PLY, tell me the path.")
