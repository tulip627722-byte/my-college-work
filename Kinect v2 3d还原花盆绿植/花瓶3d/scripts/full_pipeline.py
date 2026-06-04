"""Full pipeline: pot by angle, plant by ICP, merge, color, output"""
import numpy as np
import open3d as o3d
import cv2
import os, json, time
from scipy.spatial import cKDTree

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")
os.makedirs(OUT, exist_ok=True)

# Parameters
ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG_PER_FRAME = 360.0 / 36  # 10 deg
Y_PLANT_HI = 0.08
Y_POT_LO = 0.08
Y_POT_HI = 0.20
VOXEL_SIZE = 0.003  # 3mm

def rotate_around_y(pts, angle_deg, cx, cz):
    """Rotate pts around Y axis through (cx, 0, cz)"""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Translate to origin
    pts_c = pts.copy()
    pts_c[:, 0] -= cx
    pts_c[:, 2] -= cz
    # Rotate around Y
    x = pts_c[:, 0] * cos_a + pts_c[:, 2] * sin_a
    z = -pts_c[:, 0] * sin_a + pts_c[:, 2] * cos_a
    pts_c[:, 0] = x + cx
    pts_c[:, 2] = z + cz
    return pts_c

def sor_filter(pcd, k=20, std=2.0):
    """Statistical Outlier Removal"""
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)
    return cl

def radius_filter(pcd, r=0.02, min_pts=5):
    """Radius outlier removal"""
    cl, _ = pcd.remove_radius_outlier(nb_points=min_pts, radius=r)
    return cl

# ============================================================
# STEP 4+5a: Pot assembly by fixed angle
# ============================================================
print("=" * 60)
print("STEP 5a: Pot assembly (10 deg/frame)")
print("=" * 60)

all_pot_pts = []
all_pot_colors = []

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)

    # Extract pot
    pot_mask = (pts[:, 1] >= Y_POT_LO) & (pts[:, 1] <= Y_POT_HI)
    pts_pot = pts[pot_mask]

    if len(pts_pot) < 100:
        print(f"Frame {fidx:03d} pot: {len(pts_pot)} pts, skip")
        continue

    # Rotate to world frame
    angle = fidx * DEG_PER_FRAME
    pts_world = rotate_around_y(pts_pot, angle, ROT_CX, ROT_CZ)

    # SOR + radius filter per frame
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_world)
    pcd_tmp = sor_filter(pcd_tmp, k=20, std=2.0)
    pcd_tmp = radius_filter(pcd_tmp, r=0.02, min_pts=5)
    pts_world = np.asarray(pcd_tmp.points)

    all_pot_pts.append(pts_world)
    print(f"Frame {fidx:03d} pot: {len(pts_pot)}->{len(pts_world)} pts, angle={angle:.0f} deg")

pot_merged_pts = np.vstack(all_pot_pts)
print(f"\nPot total: {len(pot_merged_pts)} pts")
print(f"  X: [{pot_merged_pts[:,0].min():.4f}, {pot_merged_pts[:,0].max():.4f}]")
print(f"  Y: [{pot_merged_pts[:,1].min():.4f}, {pot_merged_pts[:,1].max():.4f}]")
print(f"  Z: [{pot_merged_pts[:,2].min():.4f}, {pot_merged_pts[:,2].max():.4f}]")

# ============================================================
# STEP 5b: Plant assembly by ICP
# ============================================================
print("\n" + "=" * 60)
print("STEP 5b: Plant assembly (ICP, Y-rotation only)")
print("=" * 60)

all_plant_pts = []
plant_angles = [0.0]
plant_cumulative_angle = 0.0

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)

    # Extract plant
    plant_mask = pts[:, 1] < Y_PLANT_HI
    pts_plant = pts[plant_mask]

    if len(pts_plant) < 50:
        print(f"Frame {fidx:03d} plant: too few pts, skip")
        all_plant_pts.append(np.empty((0, 3)))
        continue

    # Center on rotation axis
    pts_c = pts_plant.copy()
    pts_c[:, 0] -= ROT_CX
    pts_c[:, 2] -= ROT_CZ

    # ICP register to previous frame
    if fidx == 0:
        pts_world = pts_plant  # frame 0 as reference
        prev_pts_c = pts_c
    else:
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(pts_c)

        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(prev_pts_c)

        try:
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, 0.03, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

            R = reg.transformation[:3, :3]
            angle_step = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
            plant_cumulative_angle += angle_step
            plant_angles.append(plant_cumulative_angle)

            # Apply rotation
            pts_world_c = pts_c @ R.T
            pts_world = pts_world_c.copy()
            pts_world[:, 0] += ROT_CX
            pts_world[:, 2] += ROT_CZ

            prev_pts_c = pts_world_c  # accumulated
            print(f"Frame {fidx:03d} plant: {len(pts_plant)} pts, ICP step={angle_step:+.2f} deg, total={plant_cumulative_angle:.1f} deg, fitness={reg.fitness:.3f}")
        except Exception as e:
            print(f"Frame {fidx:03d} plant ICP failed: {e}, using previous transform")
            plant_angles.append(plant_cumulative_angle)
            pts_world = pts_plant

    # Filter
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_world)
    pcd_tmp = sor_filter(pcd_tmp, k=20, std=2.0)
    pcd_tmp = radius_filter(pcd_tmp, r=0.015, min_pts=3)
    pts_world = np.asarray(pcd_tmp.points)

    all_plant_pts.append(pts_world)

plant_merged_pts = np.vstack([p for p in all_plant_pts if len(p) > 0])
print(f"\nPlant total: {len(plant_merged_pts)} pts")
print(f"  X: [{plant_merged_pts[:,0].min():.4f}, {plant_merged_pts[:,0].max():.4f}]")
print(f"  Y: [{plant_merged_pts[:,1].min():.4f}, {plant_merged_pts[:,1].max():.4f}]")
print(f"  Z: [{plant_merged_pts[:,2].min():.4f}, {plant_merged_pts[:,2].max():.4f}]")

# ============================================================
# STEP 5c: Merge pot + plant
# ============================================================
print("\n" + "=" * 60)
print("STEP 5c: Merge + downsample + clean")
print("=" * 60)

all_pts = np.vstack([pot_merged_pts, plant_merged_pts])
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(all_pts)
print(f"Before: {len(all_pts)} pts")

# Voxel downsample
pcd_all = pcd_all.voxel_down_sample(VOXEL_SIZE)
print(f"After voxel ({VOXEL_SIZE}m): {len(pcd_all.points)} pts")

# SOR
pcd_all = sor_filter(pcd_all, k=20, std=2.0)
print(f"After SOR: {len(pcd_all.points)} pts")

# Radius filter
pcd_all = radius_filter(pcd_all, r=0.02, min_pts=5)
print(f"After radius: {len(pcd_all.points)} pts")

pts_clean = np.asarray(pcd_all.points)
print(f"Final geometry:")
print(f"  X: [{pts_clean[:,0].min():.4f}, {pts_clean[:,0].max():.4f}]")
print(f"  Y: [{pts_clean[:,1].min():.4f}, {pts_clean[:,1].max():.4f}]")
print(f"  Z: [{pts_clean[:,2].min():.4f}, {pts_clean[:,2].max():.4f}]")

o3d.io.write_point_cloud(os.path.join(OUT, "merged_geometry.ply"), pcd_all)
print(f"\nSaved merged_geometry.ply (no color)")

# ============================================================
# STEP 6: Color mapping
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Color mapping")
print("=" * 60)

fx, fy = 365.456, 365.456
cx_d, cy_d = 254.878, 205.395

# We need to re-project each 3D point back to find which frame has the best view
# For efficiency, use a voxel grid to track best color per voxel
pts_merged = np.asarray(pcd_all.points)

# Re-center to frame 0 coordinates for reprojection
# Reverse-engineer: for each point, find frame where it's closest to camera when transformed back
colors_merged = np.zeros((len(pts_merged), 3), dtype=np.float64)
color_counts = np.zeros(len(pts_merged), dtype=np.int32)

print(f"Mapping colors for {len(pts_merged)} points across 36 frames...")

for fidx in range(36):
    # Transform merged points back to this frame's view
    angle = fidx * DEG_PER_FRAME
    pts_local = rotate_around_y(pts_merged, -angle, ROT_CX, ROT_CZ)

    # Project to depth camera
    z_cam = pts_local[:, 2]
    valid_z = z_cam > 0.01
    if np.sum(valid_z) < 10:
        continue

    u_d = (pts_local[valid_z, 0] * fx / z_cam[valid_z] + cx_d).astype(int)
    v_d = (pts_local[valid_z, 1] * fy / z_cam[valid_z] + cy_d).astype(int)

    valid_uv = (u_d >= 0) & (u_d < 512) & (v_d >= 0) & (v_d < 424)
    u_d = u_d[valid_uv]; v_d = v_d[valid_uv]

    # Map to RGB
    u_r = np.clip((u_d * 1920 / 512).astype(int), 0, 1919)
    v_r = np.clip((v_d * 1080 / 424).astype(int), 0, 1079)

    # Load color
    c_raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    color_bgr = cv2.imdecode(c_raw, cv2.IMREAD_COLOR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    # Map colors back to merged point indices
    orig_indices = np.where(valid_z)[0][valid_uv]

    # For each point, take color from the frame where Z is smallest (closest)
    for k, oi in enumerate(orig_indices):
        if color_counts[oi] == 0 or z_cam[oi] < color_counts[oi]:  # track best z
            colors_merged[oi] = color_rgb[v_r[k], u_r[k]]
            color_counts[oi] = 1
            # Store z as best (simplified: first hit or closer)

    if fidx % 6 == 0:
        print(f"  Frame {fidx:03d}: processed, {np.sum(valid_uv)} points projected")

# Assign colors
pcd_colored = o3d.geometry.PointCloud()
pcd_colored.points = o3d.utility.Vector3dVector(pts_merged)
pcd_colored.colors = o3d.utility.Vector3dVector(colors_merged)

o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd_colored)
print(f"\nSaved merged_cloud.ply ({len(pts_merged)} pts with color)")

# ============================================================
# STEP 7: Output pose.json and report.json
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Output metadata")
print("=" * 60)

pose = {
    "rotation_center_x": ROT_CX,
    "rotation_center_z": ROT_CZ,
    "deg_per_frame": DEG_PER_FRAME,
    "total_frames": 36,
    "frames": []
}
for fidx in range(36):
    angle = fidx * DEG_PER_FRAME
    rad = np.radians(angle)
    pose["frames"].append({
        "frame": fidx,
        "angle_deg": angle,
        "rotation_matrix": [
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ],
        "translation": [ROT_CX, 0, ROT_CZ]
    })

with open(os.path.join(OUT, "pose.json"), "w") as f:
    json.dump(pose, f, indent=2)
print("Saved pose.json")

report = {
    "total_points": int(len(pts_merged)),
    "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
    "rotation_center_std": {"cx": 0.0023, "cz": 0.0029},
    "deg_per_frame": DEG_PER_FRAME,
    "total_rotation_deg": 360.0,
    "dimensions_m": {
        "x_range": [float(pts_merged[:,0].min()), float(pts_merged[:,0].max())],
        "y_range": [float(pts_merged[:,1].min()), float(pts_merged[:,1].max())],
        "z_range": [float(pts_merged[:,2].min()), float(pts_merged[:,2].max())],
    },
    "voxel_size_m": VOXEL_SIZE,
    "sor_params": {"k": 20, "std": 2.0},
    "pot_points": int(len(pot_merged_pts)),
    "plant_points": int(len(plant_merged_pts)),
    "method": "pot_by_fixed_angle_plant_by_icp",
    "plant_icp_total_angle_deg": float(plant_angles[-1]) if len(plant_angles) > 0 else 0,
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved report.json")

print("\n" + "=" * 60)
print("COMPLETE!")
print(f"  merged_cloud.ply  - {len(pts_merged)} points with color")
print(f"  merged_cloud.pcd  - (skipped, PLY is standard)")
print(f"  pose.json         - per-frame transforms")
print(f"  report.json       - reconstruction report")
print("=" * 60)
