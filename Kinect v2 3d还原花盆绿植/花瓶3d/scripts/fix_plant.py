"""Fix: use same 10 deg/frame for both pot AND plant, redo merge + color"""
import numpy as np
import open3d as o3d
import cv2, os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG = 360.0 / 36
VOXEL = 0.003
Y_PLANT_HI = 0.08
Y_POT_LO, Y_POT_HI = 0.08, 0.20

def rot_y(pts, deg, cx, cz):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    p = pts.copy(); p[:,0] -= cx; p[:,2] -= cz
    x = p[:,0]*c + p[:,2]*s; z = -p[:,0]*s + p[:,2]*c
    p[:,0] = x + cx; p[:,2] = z + cz
    return p

def sor(pcd, k=20, std=2.0):
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)
    return cl

def rfilter(pcd, r=0.02, m=5):
    cl, _ = pcd.remove_radius_outlier(nb_points=m, radius=r)
    return cl

# ---- Assemble all frames with uniform rotation ----
all_pts = []
print(f"Assembling 36 frames with {DEG:.1f} deg/frame...")

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)

    # Rotate entire frame by same angle
    angle = fidx * DEG
    pts_w = rot_y(pts, angle, ROT_CX, ROT_CZ)

    # Per-frame filtering
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(pts_w)
    tmp = sor(tmp, k=20, std=2.5)
    tmp = rfilter(tmp, r=0.02, m=5)
    pts_w = np.asarray(tmp.points)

    all_pts.append(pts_w)

all_merged = np.vstack(all_pts)
print(f"Before voxel: {len(all_merged)} pts")

pcd_m = o3d.geometry.PointCloud()
pcd_m.points = o3d.utility.Vector3dVector(all_merged)
pcd_m = pcd_m.voxel_down_sample(VOXEL)
print(f"After voxel: {len(pcd_m.points)} pts")
pcd_m = sor(pcd_m, k=20, std=2.0)
pcd_m = rfilter(pcd_m, r=0.02, m=5)
print(f"After clean: {len(pcd_m.points)} pts")

pts_final = np.asarray(pcd_m.points)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_geometry.ply"), pcd_m)
print(f"\nSaved merged_geometry.ply")

# ---- Color mapping ----
print(f"\nColor mapping {len(pts_final)} points...")
FX, FY = 365.456, 365.456
CXD, CYD = 254.878, 205.395

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
    print(f"  Frame {fidx:03d}: {len(idx)} visible, {len(ui)} updated")

miss = best_z == np.inf; colors[miss] = [0.6, 0.6, 0.6]
print(f"Uncolored: {np.sum(miss)}")

pcd_c = o3d.geometry.PointCloud()
pcd_c.points = o3d.utility.Vector3dVector(pts_final)
pcd_c.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd_c)
print(f"Saved merged_cloud.ply")

# Update report
report = {
    "total_points": len(pts_final), "rotation_center": {"cx": ROT_CX, "cz": ROT_CZ},
    "deg_per_frame": DEG, "total_rotation_deg": 360.0,
    "dimensions_m": {"x": [float(pts_final[:,0].min()), float(pts_final[:,0].max())],
                       "y": [float(pts_final[:,1].min()), float(pts_final[:,1].max())],
                       "z": [float(pts_final[:,2].min()), float(pts_final[:,2].max())]},
    "method": "uniform_rotation_pot_and_plant"
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved report.json\nDone.")
