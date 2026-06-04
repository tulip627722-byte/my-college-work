"""Step 2+3: DBSCAN cleaning + RANSAC circle fitting for rotation axis"""
import numpy as np
import cv2
import open3d as o3d
import os
from collections import Counter

DATA = r"E:\花瓶3d"
CROP = os.path.join(DATA, "output", "cropped_frames")
OUT = os.path.join(DATA, "output")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# STEP 2: DBSCAN on frame 0 — clean single frame
# ============================================================
print("=" * 60)
print("STEP 2: DBSCAN Clustering on Frame 0")
print("=" * 60)

pcd0 = o3d.io.read_point_cloud(os.path.join(CROP, "frame_000_crop.ply"))
pts0 = np.asarray(pcd0.points)
print(f"Input: {len(pts0)} points")

# DBSCAN
labels = np.array(pcd0.cluster_dbscan(eps=0.02, min_points=30, print_progress=False))
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN found {n_clusters} clusters, noise: {np.sum(labels == -1)}")

# Cluster stats
cluster_counts = Counter(labels[labels >= 0])
largest_label = cluster_counts.most_common(1)[0][0]
print(f"Largest cluster: label {largest_label}, {cluster_counts[largest_label]} points")
for lbl, cnt in cluster_counts.most_common(5):
    print(f"  Cluster {lbl}: {cnt} pts")

# Keep largest cluster only
keep = labels == largest_label
pts_clean = pts0[keep]
colors_clean = np.asarray(pcd0.colors)[keep] if pcd0.has_colors() else None

pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(pts_clean)
if colors_clean is not None:
    pcd_clean.colors = o3d.utility.Vector3dVector(colors_clean)

o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_clean.ply"), pcd_clean)
print(f"\nStep 2 Result:")
print(f"  Clean points: {len(pts_clean)}")
print(f"  X: [{pts_clean[:,0].min():.4f}, {pts_clean[:,0].max():.4f}]")
print(f"  Y: [{pts_clean[:,1].min():.4f}, {pts_clean[:,1].max():.4f}]")
print(f"  Z: [{pts_clean[:,2].min():.4f}, {pts_clean[:,2].max():.4f}]")
print(f"  Saved: debug_frame000_clean.ply")

# ============================================================
# STEP 3: RANSAC circle fitting on XZ plane for all frames
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: XZ-plane RANSAC Circle Fitting (Rotation Axis)")
print("=" * 60)

def fit_circle_ransac_2d(pts_xz, n_iter=500, thresh=0.005):
    """Fit circle in 2D using RANSAC. pts_xz: (N,2) array of (x,z) coords."""
    best_inliers = 0
    best_cx, best_cz, best_r = 0, 0, 0

    for _ in range(n_iter):
        # Sample 3 random points
        idx = np.random.choice(len(pts_xz), 3, replace=False)
        p1, p2, p3 = pts_xz[idx]

        # Solve circle from 3 points
        # Using perpendicular bisector intersection
        mid12 = (p1 + p2) / 2
        mid23 = (p2 + p3) / 2
        d12 = p2 - p1  # (dx, dz)
        d23 = p3 - p2

        # Perpendicular directions
        n12 = np.array([-d12[1], d12[0]])
        n23 = np.array([-d23[1], d23[0]])

        # Solve: mid12 + t*n12 = mid23 + s*n23
        A = np.column_stack([n12, -n23])
        b = mid23 - mid12

        try:
            ts = np.linalg.solve(A, b)
            cx, cz = mid12 + ts[0] * n12
            r = np.linalg.norm(p1 - np.array([cx, cz]))
        except np.linalg.LinAlgError:
            continue

        if r <= 0 or r > 1.0:
            continue

        # Count inliers
        dists = np.abs(np.sqrt((pts_xz[:,0] - cx)**2 + (pts_xz[:,1] - cz)**2) - r)
        inliers = np.sum(dists < thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_cx, best_cz, best_r = cx, cz, r

    inlier_ratio = best_inliers / len(pts_xz)
    return best_cx, best_cz, best_r, best_inliers, inlier_ratio

centers = []
for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CROP, f"frame_{fidx:03d}_crop.ply"))
    pts = np.asarray(pcd.points)

    # DBSCAN clean
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=30, print_progress=False))
    if len(set(labels)) <= 1:
        print(f"Frame {fidx:03d}: DBSCAN failed (<2 clusters), using all points")
        pts_clean_f = pts
    else:
        cnt = Counter(labels[labels >= 0])
        if cnt:
            best_label = cnt.most_common(1)[0][0]
            pts_clean_f = pts[labels == best_label]
        else:
            pts_clean_f = pts

    # XZ projection for circle fitting
    xz = pts_clean_f[:, [0, 2]]  # X and Z columns

    cx, cz, r, ninliers, ratio = fit_circle_ransac_2d(xz, n_iter=500, thresh=0.005)

    # Also compute simple centroid
    cx_mean = np.mean(pts_clean_f[:, 0])
    cz_mean = np.mean(pts_clean_f[:, 2])

    centers.append((cx, cz, r, ninliers, ratio, len(pts_clean_f), cx_mean, cz_mean))
    print(f"Frame {fidx:03d}: center=({cx:.4f},{cz:.4f}) r={r:.4f} inliers={ninliers}/{len(pts_clean_f)} ({ratio:.2f}) "
          f"pts={len(pts_clean_f)}")

# Analyze centers
cxs = np.array([c[0] for c in centers])
czs = np.array([c[1] for c in centers])
rs = np.array([c[2] for c in centers])
ratios = np.array([c[4] for c in centers])

print(f"\n{'='*60}")
print(f"STEP 3 Results — Circle Centers Distribution:")
print(f"  CX: mean={cxs.mean():.4f}, std={cxs.std():.4f}, median={np.median(cxs):.4f}")
print(f"  CZ: mean={czs.mean():.4f}, std={czs.std():.4f}, median={np.median(czs):.4f}")
print(f"  R:  mean={rs.mean():.4f}, std={rs.std():.4f}, median={np.median(rs):.4f}")
print(f"  Inlier ratio: mean={ratios.mean():.4f}, min={ratios.min():.4f}, max={ratios.max():.4f}")

# Use median as rotation center (more robust)
cx_med = np.median(cxs)
cz_med = np.median(czs)
r_med = np.median(rs)

# Also try mean centroid approach for comparison
cx_centroid_mean = np.mean([c[6] for c in centers])
cz_centroid_mean = np.mean([c[7] for c in centers])

print(f"\n  Rotation center (RANSAC median):   ({cx_med:.4f}, {cz_med:.4f})")
print(f"  Rotation center (centroid mean):   ({cx_centroid_mean:.4f}, {cz_centroid_mean:.4f})")
print(f"  Circle radius median: {r_med:.4f} m")

# Save
import json
results = {
    "rotation_center_x": float(cx_med),
    "rotation_center_z": float(cz_med),
    "circle_radius_m": float(r_med),
    "cx_std": float(cxs.std()),
    "cz_std": float(czs.std()),
    "per_frame": [{"frame": i, "cx": float(c[0]), "cz": float(c[1]), "r": float(c[2]),
                    "inlier_ratio": float(c[4]), "points": int(c[5])} for i, c in enumerate(centers)]
}
with open(os.path.join(OUT, "step3_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nStep 2+3 complete. Rotation center saved to step3_results.json")
print(f"Saved clean frame 0: debug_frame000_clean.ply")
