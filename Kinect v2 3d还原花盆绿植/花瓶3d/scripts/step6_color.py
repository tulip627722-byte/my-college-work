"""Step 6: Color mapping (vectorized) and Step 7: output metadata"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG_PER_FRAME = 360.0 / 36
FX, FY = 365.456, 365.456
CX_D, CY_D = 254.878, 205.395

def rotate_around_y(pts, angle_deg, cx, cz):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    p = pts.copy()
    p[:,0] -= cx; p[:,2] -= cz
    x = p[:,0]*c + p[:,2]*s
    z = -p[:,0]*s + p[:,2]*c
    p[:,0] = x + cx; p[:,2] = z + cz
    return p

# Load merged geometry
pcd = o3d.io.read_point_cloud(os.path.join(OUT, "merged_geometry.ply"))
pts = np.asarray(pcd.points)
N = len(pts)
print(f"Points: {N}")

colors = np.full((N, 3), 0.5, dtype=np.float32)
best_z = np.full(N, np.inf, dtype=np.float32)

for fidx in range(36):
    angle = fidx * DEG_PER_FRAME
    pts_local = rotate_around_y(pts, -angle, ROT_CX, ROT_CZ)
    z_cam = pts_local[:, 2].astype(np.float32)

    # Project to depth image
    u_d = np.round(pts_local[:,0] * FX / z_cam + CX_D).astype(np.int32)
    v_d = np.round(pts_local[:,1] * FY / z_cam + CY_D).astype(np.int32)

    in_bounds = (z_cam > 0.01) & (u_d >= 0) & (u_d < 512) & (v_d >= 0) & (v_d < 424)
    idx = np.where(in_bounds)[0]
    if len(idx) < 10:
        continue

    u_r = np.clip((u_d[idx] * 1920 // 512).astype(np.int32), 0, 1919)
    v_r = np.clip((v_d[idx] * 1080 // 424).astype(np.int32), 0, 1079)

    # Load RGB (uint8)
    c_raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Vectorized: where this frame is closer, update color
    z_this = z_cam[idx]
    update = z_this < best_z[idx]
    up_idx = idx[update]
    if len(up_idx) > 0:
        best_z[up_idx] = z_this[update]
        colors[up_idx] = rgb[v_r[update], u_r[update]].astype(np.float32) / 255.0

    print(f"Frame {fidx:03d}: {len(idx)} in view, {len(up_idx)} updated, {np.sum(best_z < np.inf)} colored")

# Uncolored -> gray
miss = best_z == np.inf
if np.sum(miss) > 0:
    colors[miss] = [0.6, 0.6, 0.6]
    print(f"Uncolored: {np.sum(miss)}")

pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(pts)
pcd_out.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd_out)
print(f"\nSaved merged_cloud.ply")

# ============================================================
# Step 7: Output pose.json and report.json
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Output metadata")
print("=" * 60)

pose = {"rotation_center_x": ROT_CX, "rotation_center_z": ROT_CZ,
        "deg_per_frame": DEG_PER_FRAME, "total_frames": 36, "frames": []}
for fidx in range(36):
    rad = np.radians(fidx * DEG_PER_FRAME)
    c, s = np.cos(rad), np.sin(rad)
    pose["frames"].append({
        "frame": fidx, "angle_deg": fidx * DEG_PER_FRAME,
        "rotation_matrix": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "translation": [ROT_CX, 0, ROT_CZ]
    })
with open(os.path.join(OUT, "pose.json"), "w") as f:
    json.dump(pose, f, indent=2)
print("Saved pose.json")

report = {
    "total_points": N,
    "rotation_center": {"cx": ROT_CX, "cz": ROT_CZ},
    "rotation_center_std_m": {"cx": 0.0023, "cz": 0.0029},
    "deg_per_frame": DEG_PER_FRAME,
    "total_rotation_deg": 360.0,
    "dimensions_m": {
        "x": [float(pts[:,0].min()), float(pts[:,0].max())],
        "y": [float(pts[:,1].min()), float(pts[:,1].max())],
        "z": [float(pts[:,2].min()), float(pts[:,2].max())],
    },
    "voxel_size_m": 0.003,
    "pipeline": "pot_by_angle_plant_by_icp",
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved report.json")
print("\nDONE. Files in output/: merged_cloud.ply, pose.json, report.json")
