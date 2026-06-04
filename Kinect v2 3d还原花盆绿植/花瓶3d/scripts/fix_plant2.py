"""
Fix plant: extract plant points from FULL depth frame (not 2D mask).
Use green color ratio + depth range to identify plant, keep all leaf points.
"""
import numpy as np
import cv2, open3d as o3d, os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")
os.makedirs(os.path.join(OUT, "clean_frames_v2"), exist_ok=True)

ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG = 360.0 / 36
FX, FY = 365.456, 365.456
CXD, CYD = 254.878, 205.395

# Load 2D mask for pot only
mask_raw = np.fromfile(os.path.join(OUT, "mask_2d.png"), dtype=np.uint8)
mask_2d = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)

def rot_y(pts, deg, cx, cz):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    p = pts.copy(); p[:,0] -= cx; p[:,2] -= cz
    x = p[:,0]*c + p[:,2]*s; z = -p[:,0]*s + p[:,2]*c
    p[:,0] = x + cx; p[:,2] = z + cz
    return p

all_pts, all_colors = [], []

for fidx in range(36):
    # ---- Load full depth + color ----
    d_raw = np.fromfile(os.path.join(DATA, "depth", f"{fidx:03d}.png"), dtype=np.uint8)
    depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
    h, w = depth.shape

    c_raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    color_rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # ---- Generate full point cloud ----
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) / 1000.0
    valid = (z > 0.3) & (z < 2.5)  # wider depth range for leaves
    zv, uv, vv = z[valid], u[valid], v[valid]
    x = (uv - CXD) * zv / FX
    y = (vv - CYD) * zv / FY
    pts_full = np.stack([x, y, zv], axis=1)

    # Map to RGB for color info
    ur = np.clip((uv * 1920 // 512).astype(int), 0, 1919)
    vr = np.clip((vv * 1080 // 424).astype(int), 0, 1079)
    clr_full = color_rgb[vr, ur]

    # ---- POT: use 2D mask + Y range ----
    in_mask = mask_2d[vr, ur] > 0
    is_pot = in_mask & (y >= 0.08) & (y <= 0.20) & (zv > 0.4) & (zv < 2.0)
    pts_pot = pts_full[is_pot]
    clr_pot = clr_full[is_pot]

    # ---- PLANT: use GREEN color + Y range (NO 2D mask restriction) ----
    r, g, b = clr_full[:, 0].astype(float), clr_full[:, 1].astype(float), clr_full[:, 2].astype(float)
    green_ratio = g / (r + g + b + 0.001)
    is_green = (green_ratio > 0.36) & (g > r * 0.9) & (g > b * 0.9)
    is_plant = is_green & (y < 0.08) & (y > -0.1) & (zv > 0.4) & (zv < 2.5)
    pts_plant = pts_full[is_plant]
    clr_plant = clr_full[is_plant]

    print(f"Frame {fidx:03d}: pot={len(pts_pot)} plant={len(pts_plant)}")

    if len(pts_pot) < 50 and len(pts_plant) < 50:
        continue

    # Combine pot + plant
    pts_frame = np.vstack([pts_pot, pts_plant]) if len(pts_pot) > 0 and len(pts_plant) > 0 else \
                pts_pot if len(pts_plant) == 0 else pts_plant
    clr_frame = np.vstack([clr_pot, clr_plant]) if len(clr_pot) > 0 and len(clr_plant) > 0 else \
                clr_pot if len(clr_plant) == 0 else clr_plant

    # Rotate to world
    angle = fidx * DEG
    pts_w = rot_y(pts_frame, angle, ROT_CX, ROT_CZ)

    # Very light per-frame filtering for plant, standard for pot
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_w)
    pcd_tmp, _ = pcd_tmp.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.0)
    pcd_tmp, _ = pcd_tmp.remove_radius_outlier(nb_points=3, radius=0.015)
    pts_w = np.asarray(pcd_tmp.points)

    # Keep track of colors for merge (dedup later)
    all_pts.append(pts_w)

all_merged = np.vstack(all_pts)
print(f"\nPre-merge: {len(all_merged)} pts")

pcd_m = o3d.geometry.PointCloud()
pcd_m.points = o3d.utility.Vector3dVector(all_merged)
pcd_m = pcd_m.voxel_down_sample(0.003)
print(f"After voxel: {len(pcd_m.points)}")
pcd_m, _ = pcd_m.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
print(f"After SOR: {len(pcd_m.points)}")

pts_final = np.asarray(pcd_m.points)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_geometry.ply"), pcd_m)

# ---- Color mapping ----
print(f"\nColor mapping {len(pts_final)} points...")
colors = np.full((len(pts_final), 3), 0.5, dtype=np.float32)
best_z = np.full(len(pts_final), np.inf, dtype=np.float32)

for fidx in range(36):
    angle = fidx * DEG
    pts_l = rot_y(pts_final, -angle, ROT_CX, ROT_CZ)
    zc = pts_l[:, 2].astype(np.float32)
    ud = np.round(pts_l[:,0]*FX/zc + CXD).astype(np.int32)
    vd = np.round(pts_l[:,1]*FY/zc + CYD).astype(np.int32)
    ok = (zc > 0.01) & (ud >= 0) & (ud < 512) & (vd >= 0) & (vd < 424)
    idx = np.where(ok)[0]
    if len(idx) < 10: continue

    ur = np.clip((ud[idx] * 1920 // 512).astype(np.int32), 0, 1919)
    vr = np.clip((vd[idx] * 1080 // 424).astype(np.int32), 0, 1079)

    raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    zt = zc[idx]; upd = zt < best_z[idx]; ui = idx[upd]
    if len(ui):
        best_z[ui] = zt[upd]
        colors[ui] = rgb[vr[upd], ur[upd]].astype(np.float32) / 255.0
    if fidx % 6 == 0:
        print(f"  Frame {fidx:03d}: {len(idx)} visible, {len(ui)} updated")

miss = best_z == np.inf; colors[miss] = [0.5, 0.5, 0.5]

pcd_c = o3d.geometry.PointCloud()
pcd_c.points = o3d.utility.Vector3dVector(pts_final)
pcd_c.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd_c)
print(f"Saved merged_cloud.ply ({len(pts_final)} pts, {np.sum(miss)} uncolored)")

# Report
pts_f = pts_final
report = {
    "total_points": len(pts_f), "rotation_center": {"cx": ROT_CX, "cz": ROT_CZ},
    "deg_per_frame": DEG, "total_rotation_deg": 360.0,
    "dimensions_m": {"x": [float(pts_f[:,0].min()), float(pts_f[:,0].max())],
                       "y": [float(pts_f[:,1].min()), float(pts_f[:,1].max())],
                       "z": [float(pts_f[:,2].min()), float(pts_f[:,2].max())]},
    "method": "pot_by_mask+plant_by_green_ratio",
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Done.")
