"""Separate pot/plant by Y threshold, fit rotation axis from pot"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CROP = os.path.join(DATA, "output", "cropped_frames")
OUT = os.path.join(DATA, "output")

Y_PLANT_LO = -0.1   # plant lower bound
Y_POT_LO = 0.08      # pot lower bound
Y_POT_HI = 0.20      # pot upper bound (above = discard)

pot_centroids = []
all_stats = []

# save path
clean_dir = os.path.join(OUT, "clean_frames")
os.makedirs(clean_dir, exist_ok=True)

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CROP, f"frame_{fidx:03d}_crop.ply"))
    pts = np.asarray(pcd.points)

    y = pts[:, 1]
    is_plant = y < Y_POT_LO
    is_pot = (y >= Y_POT_LO) & (y <= Y_POT_HI)

    pts_pot = pts[is_pot]
    pts_plant = pts[is_plant]

    # DBSCAN clean pot
    if len(pts_pot) > 50:
        pcd_pot = o3d.geometry.PointCloud()
        pcd_pot.points = o3d.utility.Vector3dVector(pts_pot)
        labels = np.array(pcd_pot.cluster_dbscan(eps=0.02, min_points=30, print_progress=False))
        # Take largest cluster
        from collections import Counter
        cnt = Counter(labels[labels >= 0])
        if cnt:
            best_lbl = cnt.most_common(1)[0][0]
            pts_pot = pts_pot[labels == best_lbl]

    # Keep all plant points (no DBSCAN needed)
    # Combine
    pts_keep = np.vstack([pts_pot, pts_plant]) if len(pts_pot) > 0 and len(pts_plant) > 0 else \
               pts_pot if len(pts_plant) == 0 else pts_plant

    # Save cleaned frame (colors re-applied in Step 6)
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_keep)
    o3d.io.write_point_cloud(
        os.path.join(OUT, "clean_frames", f"frame_{fidx:03d}_clean.ply"), pcd_out)

    # Pot centroid
    if len(pts_pot) > 0:
        cx, cz = np.mean(pts_pot[:, 0]), np.mean(pts_pot[:, 2])
        pot_centroids.append((fidx, cx, cz, len(pts_pot)))

    all_stats.append((fidx, len(pts_pot), len(pts_plant), len(pts_keep)))
    print(f"Frame {fidx:03d}: pot={len(pts_pot):4d}  plant={len(pts_plant):4d}  keep={len(pts_keep):4d}")

# ============================================================
# Rotation axis from pot centroids
# ============================================================
os.makedirs(os.path.join(OUT, "clean_frames"), exist_ok=True)

cxs = np.array([c[1] for c in pot_centroids])
czs = np.array([c[2] for c in pot_centroids])

print(f"\n{'='*60}")
print(f"Pot centroid distribution (36 frames):")
print(f"  CX: mean={cxs.mean():.4f}, std={cxs.std():.4f}, median={np.median(cxs):.4f}")
print(f"  CZ: mean={czs.mean():.4f}, std={czs.std():.4f}, median={np.median(czs):.4f}")

rot_cx = np.median(cxs)
rot_cz = np.median(czs)
dists = np.sqrt((cxs - rot_cx)**2 + (czs - rot_cz)**2)
print(f"  Rotation center: ({rot_cx:.4f}, {rot_cz:.4f})")
print(f"  Distance from center: mean={dists.mean():.4f}m, max={dists.max():.4f}m")

rotation = {
    "rotation_center_x": float(rot_cx),
    "rotation_center_z": float(rot_cz),
    "cx_std": float(cxs.std()),
    "cz_std": float(czs.std()),
}
with open(os.path.join(OUT, "rotation_axis.json"), "w") as f:
    json.dump(rotation, f, indent=2)

print(f"\nSaved rotation_axis.json")
print(f"Clean frames saved to clean_frames/")
print(f"\nReady for Step 4 — Phase angle estimation.")
