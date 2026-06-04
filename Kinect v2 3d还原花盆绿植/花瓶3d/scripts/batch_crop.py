"""Batch crop all 36 frames using 2D mask + depth filter"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output", "cropped_frames")
os.makedirs(OUT, exist_ok=True)

# Load & dilate mask (1920x1080)
mask_raw = np.fromfile(os.path.join(DATA, "output", "mask_2d.png"), dtype=np.uint8)
mask_rgb = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=1)
print(f"Mask dilated: {np.sum(mask_rgb > 0)} active pixels (+{(np.sum(mask_rgb>0)-61579)/61579*100:.0f}%)")

fx = fy = 365.456
cx, cy = 254.878, 205.395

for fidx in range(36):
    # Depth
    d_raw = np.fromfile(os.path.join(DATA, "depth", f"{fidx:03d}.png"), dtype=np.uint8)
    depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
    h, w = depth.shape

    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0.001
    zv = z[valid]; uv = u[valid]; vv = v[valid]

    # Map to RGB mask
    u_rgb = np.clip((uv * 1920 / 512).astype(int), 0, 1919)
    v_rgb = np.clip((vv * 1080 / 424).astype(int), 0, 1079)
    in_mask = mask_rgb[v_rgb, u_rgb] > 0

    zv = zv[in_mask]; uv = uv[in_mask]; vv = vv[in_mask]
    u_rgb = u_rgb[in_mask]; v_rgb = v_rgb[in_mask]

    if len(zv) == 0:
        print(f"Frame {fidx:03d}: SKIP (0 points after mask)")
        continue

    x = (uv - cx) * zv / fx
    y = (vv - cy) * zv / fy
    pts = np.stack([x, y, zv], axis=1)

    # Depth filter 0.4~2.0m
    depth_mask_ok = (zv > 0.4) & (zv < 2.0)
    pts = pts[depth_mask_ok]
    u_rgb = u_rgb[depth_mask_ok]
    v_rgb = v_rgb[depth_mask_ok]

    if len(pts) < 100:
        print(f"Frame {fidx:03d}: only {len(pts)} pts after depth filter, skip")
        continue

    # Color
    c_raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    color_rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    colors = color_rgb[v_rgb, u_rgb] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(OUT, f"frame_{fidx:03d}_crop.ply"), pcd)
    print(f"Frame {fidx:03d}: {len(pts):5d} pts, X=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}], Y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}], Z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

print(f"\nDone. Cropped frames in: {OUT}")
print(f"Now ready for Step 2+3 (cleaning + cylinder fitting).")
