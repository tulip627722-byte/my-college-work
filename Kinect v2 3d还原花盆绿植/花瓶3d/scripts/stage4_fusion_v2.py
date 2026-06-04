"""
Stage 4 v2: Merge point clouds first, then ONE unified Poisson reconstruction
Pot: rot_center transforms (direct 10deg)
Plant: ICP transforms (v2 incremental)
"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG = 360.0 / 36

print("=" * 60)
print("STAGE 4 v2: Unified point cloud + Poisson")
print("=" * 60)

# ============================================================
# Load existing world-coordinate point clouds
# ============================================================
print("\nLoading point clouds...")
pot_pcd = o3d.io.read_point_cloud(os.path.join(OUT, "pot_merged_color.ply"))
plant_pcd = o3d.io.read_point_cloud(os.path.join(OUT, "plant_merged.ply"))
print(f"  Pot:   {len(pot_pcd.points)} pts")
print(f"  Plant: {len(plant_pcd.points)} pts")

# ============================================================
# Merge into one unified point cloud
# ============================================================
print("\nMerging point clouds...")
pot_pts = np.asarray(pot_pcd.points)
plant_pts = np.asarray(plant_pcd.points)
pot_clr = np.asarray(pot_pcd.colors) if pot_pcd.has_colors() else None
plant_clr = np.asarray(plant_pcd.colors) if plant_pcd.has_colors() else None

# Check if plant has colors
if plant_clr is None or len(plant_clr) == 0:
    plant_clr = np.full_like(plant_pts, [0.2, 0.7, 0.3])  # green default

merged_pts = np.vstack([pot_pts, plant_pts])
merged_clr = np.vstack([pot_clr, plant_clr])
print(f"  Combined: {len(merged_pts)} pts")

unified = o3d.geometry.PointCloud()
unified.points = o3d.utility.Vector3dVector(merged_pts)
unified.colors = o3d.utility.Vector3dVector(merged_clr)

# Downsample to reasonable density
unified = unified.voxel_down_sample(0.002)
print(f"  After voxel(2mm): {len(unified.points)} pts")

unified, _ = unified.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
print(f"  After SOR: {len(unified.points)} pts")

# ============================================================
# Save merged point cloud
# ============================================================
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), unified)
# PCD as ASCII (Chinese path workaround)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.pcd"), unified, write_ascii=True)
print("Saved: merged_cloud.ply, merged_cloud.pcd")

# ============================================================
# Estimate normals for Poisson
# ============================================================
print("\nEstimating normals...")
unified.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=40))
unified.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

# ============================================================
# ONE unified Poisson reconstruction
# ============================================================
print("Poisson reconstruction (depth=9)...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    unified, depth=9, width=0, scale=1.1, linear_fit=False)

# Remove low-density faces (outliers)
densities = np.asarray(densities)
d_min, d_max = densities.min(), densities.max()
threshold = d_min + 0.05 * (d_max - d_min)
mesh.remove_vertices_by_mask(densities < threshold)

# Cleanup
mesh = mesh.simplify_vertex_clustering(0.002, contraction=o3d.geometry.SimplificationContraction.Average)
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()

verts = np.asarray(mesh.vertices)
fin = np.all(np.isfinite(verts), axis=1)
if not fin.all():
    mesh.remove_vertices_by_mask(~fin)

# Light smoothing
mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
mesh.compute_vertex_normals()

# ============================================================
# Color transfer from unified point cloud
# ============================================================
print("Transferring colors...")
from scipy.spatial import cKDTree
pcd_pts = np.asarray(unified.points)
pcd_clr = np.asarray(unified.colors)
tree = cKDTree(pcd_pts)
mesh_verts = np.asarray(mesh.vertices)
_, nn_idx = tree.query(mesh_verts, k=1)
mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_clr[nn_idx])

print(f"\nFused mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.ply"), mesh)
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.obj"), mesh)
print("Saved: fused_mesh.ply, fused_mesh.obj")

# ============================================================
# pose.json
# ============================================================
print("\nGenerating pose.json...")
def rot_center(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    T1 = np.eye(4); T1[0,3] = -ROT_CX; T1[2,3] = -ROT_CZ
    R = np.eye(4); R[0,0]=c; R[0,2]=s; R[2,0]=-s; R[2,2]=c
    T2 = np.eye(4); T2[0,3] = ROT_CX; T2[2,3] = ROT_CZ
    return (T2 @ R @ T1).tolist()

poses = {}
for i in range(36):
    poses[f"frame_{i:03d}"] = {
        "angle_deg": round(i * DEG, 1),
        "transform_world_from_camera": rot_center(-i * DEG)
    }
with open(os.path.join(OUT, "pose.json"), "w") as f:
    json.dump(poses, f, indent=2)

# ============================================================
# report.json
# ============================================================
report = {
    "pipeline": "unified_poisson",
    "date": "2026-06-01",
    "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
    "deg_per_frame": DEG,
    "num_frames": 36,
    "pot": {"method": "direct_10deg_rotation", "source": "pot_merged_color.ply"},
    "plant": {"method": "frame0_seed_incremental_icp", "source": "plant_merged.ply"},
    "fused": {
        "method": "merge_clouds_then_single_poisson",
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "point_cloud_points": len(unified.points),
    },
    "outputs": ["fused_mesh.ply","fused_mesh.obj","merged_cloud.ply","merged_cloud.pcd","pose.json","report.json"]
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved: pose.json, report.json")

# ============================================================
# Visualize
# ============================================================
print("\n" + "=" * 60)
print("FUSED MESH — close window to finish.")
print("=" * 60)
try:
    o3d.visualization.draw_geometries(
        [mesh], window_name="Stage 4 — Fused Model (Unified Poisson)",
        width=1280, height=720, mesh_show_back_face=True)
except Exception as e:
    print(f"Viz error: {e}")

# Also show the unified point cloud
print("\nUNIFIED POINT CLOUD — close window.")
try:
    o3d.visualization.draw_geometries(
        [unified], window_name="Stage 4 — Unified Point Cloud",
        width=1280, height=720)
except Exception as e:
    print(f"Viz error: {e}")

print("\nStage 4 done! All deliverables in output_v2/")
