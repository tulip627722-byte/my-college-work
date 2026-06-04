"""
Stage 4: Final fusion — pot + plant merge, cleanup, export all deliverables
"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG = 360.0 / 36

print("=" * 60)
print("STAGE 4: Final Fusion")
print("=" * 60)

# ============================================================
# Load pot and plant meshes
# ============================================================
print("\nLoading meshes...")
pot_mesh = o3d.io.read_triangle_mesh(os.path.join(OUT, "pot_mesh_color.ply"))
plant_mesh = o3d.io.read_triangle_mesh(os.path.join(OUT, "plant_mesh.ply"))
print(f"  Pot:   {len(pot_mesh.vertices)} verts, {len(pot_mesh.triangles)} tris")
print(f"  Plant: {len(plant_mesh.vertices)} verts, {len(plant_mesh.triangles)} tris")

# ============================================================
# Merge meshes directly (both in same world coordinate system)
# ============================================================
print("\nMerging meshes...")
fused = pot_mesh + plant_mesh
fused.remove_duplicated_vertices()
fused.remove_degenerate_triangles()

# Clean non-finite
verts = np.asarray(fused.vertices)
fin = np.all(np.isfinite(verts), axis=1)
if not fin.all():
    print(f"  Removing {np.sum(~fin)} non-finite vertices")
    fused.remove_vertices_by_mask(~fin)

print(f"  Fused: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

# ============================================================
# Smoothing
# ============================================================
print("\nSmoothing...")
fused = fused.filter_smooth_taubin(number_of_iterations=5)
fused.compute_vertex_normals()
fused.remove_degenerate_triangles()
fused.remove_duplicated_vertices()
print(f"  After smooth: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

# ============================================================
# Save final mesh
# ============================================================
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.ply"), fused)
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.obj"), fused)
print("Saved: fused_mesh.ply, fused_mesh.obj")

# ============================================================
# Generate merged point cloud (PCD + PLY)
# ============================================================
print("\nGenerating merged point cloud...")
# Sample from mesh
pcd = fused.sample_points_uniformly(number_of_points=100000)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.pcd"), pcd)
print(f"Saved: merged_cloud.ply, merged_cloud.pcd ({len(pcd.points)} pts)")

# ============================================================
# Generate pose.json
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
print("Saved: pose.json")

# ============================================================
# Generate report.json
# ============================================================
print("\nGenerating report.json...")
report = {
    "pipeline": "direct_10deg_assembly",
    "date": "2026-06-01",
    "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
    "deg_per_frame": DEG,
    "num_frames": 36,
    "pot": {
        "method": "direct_10deg_rotation_no_icp",
        "mesh_vertices": len(pot_mesh.vertices),
        "mesh_triangles": len(pot_mesh.triangles),
    },
    "plant": {
        "method": "frame0_seed_incremental_icp",
        "mesh_vertices": len(plant_mesh.vertices),
        "mesh_triangles": len(plant_mesh.triangles),
    },
    "fused": {
        "mesh_vertices": len(fused.vertices),
        "mesh_triangles": len(fused.triangles),
    },
    "outputs": [
        "fused_mesh.ply", "fused_mesh.obj",
        "merged_cloud.ply", "merged_cloud.pcd",
        "pose.json", "report.json",
        "pot_mesh_color.ply", "plant_mesh.ply"
    ]
}

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved: report.json")

# ============================================================
# Visualize final result
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULT — close window to complete.")
print("=" * 60)
try:
    o3d.visualization.draw_geometries(
        [fused], window_name="Stage 4 — Final Fused Model",
        width=1280, height=720, mesh_show_back_face=True)
except Exception as e:
    print(f"Viz skipped: {e}")

# Also show wireframe + point cloud to verify alignment
print("\nAll done! Files in output_v2/:")
for f in os.listdir(OUT):
    fpath = os.path.join(OUT, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.0f} KB)")
