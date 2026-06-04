"""Step 4 v4: Brute-force angular search — rotate each frame to match frame 0"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

# Load all frames, center them on rotation axis
rot_cx, rot_cz = -0.0346, 0.7861
pcds_centered = []
for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)
    # Shift so rotation axis is at origin
    pts[:, 0] -= rot_cx
    pts[:, 2] -= rot_cz
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcds_centered.append(pcd)

ref = pcds_centered[0]
ref_pts = np.asarray(ref.points)

def rotate_pts_y(pts, angle_deg):
    """Rotate points around Y axis by angle_deg"""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, 0, sin_a],
                  [0, 1, 0],
                  [-sin_a, 0, cos_a]])
    return pts @ R.T

def chamfer_distance(pts_a, pts_b, subsample=2000):
    """Approximate Chamfer distance between two point clouds"""
    na = min(subsample, len(pts_a))
    nb = min(subsample, len(pts_b))
    ia = np.random.choice(len(pts_a), na, replace=False)
    ib = np.random.choice(len(pts_b), nb, replace=False)
    a = pts_a[ia]; b = pts_b[ib]
    # Simple: mean of min distances
    from scipy.spatial import cKDTree
    tree_b = cKDTree(b)
    dists_a_to_b, _ = tree_b.query(a)
    return np.mean(dists_a_to_b)

print("Brute-force angular search (step=1 deg, range 0-360):")
angles = [0.0]
best_angles_all = [0.0]

for fidx in range(1, 36):
    pts_src = np.asarray(pcds_centered[fidx].points)

    best_angle = 0
    best_dist = float('inf')
    prev_best = angles[-1]

    # Search around expected angle first (±30 deg), then full range
    search_range = np.concatenate([
        np.arange(prev_best - 45, prev_best + 45, 2),  # local search
        np.arange(-180, 180, 10)  # coarse global
    ])
    search_range = np.unique(np.clip(search_range, -180, 180))

    for angle in search_range:
        pts_rot = rotate_pts_y(pts_src, angle)
        d = chamfer_distance(pts_rot, ref_pts, subsample=2000)
        if d < best_dist:
            best_dist = d
            best_angle = angle

    # Fine search around best
    for angle in np.arange(best_angle - 3, best_angle + 3, 0.5):
        pts_rot = rotate_pts_y(pts_src, angle)
        d = chamfer_distance(pts_rot, ref_pts, subsample=2000)
        if d < best_dist:
            best_dist = d
            best_angle = angle

    angles.append(best_angle)
    print(f"Frame {fidx:03d}: best angle = {best_angle:+.1f} deg  (dist={best_dist:.4f})")

# Unwrap for continuity
angles_arr = np.array(angles)
# Try to make angles monotonic
for i in range(1, len(angles_arr)):
    # Add/subtract 360 to minimize jump
    diff = angles_arr[i] - angles_arr[i-1]
    if diff > 180:
        angles_arr[i] -= 360
    elif diff < -180:
        angles_arr[i] += 360

diffs = np.diff(angles_arr)
print(f"\n{'='*60}")
print(f"Brute-force results (unwrapped):")
print(f"  Adjacent steps: {[f'{d:.1f}' for d in diffs]}")
print(f"  All same sign? {'YES' if np.all(diffs > 0) or np.all(diffs < 0) else 'NO'}")
print(f"  Median step: {np.median(np.abs(diffs)):.1f} deg/frame")
print(f"  Total rotation: {abs(angles_arr[-1] - angles_arr[0]):.1f} deg")
print(f"  Range: {angles_arr[0]:.1f} to {angles_arr[-1]:.1f} deg")

# Save
pose = {
    "method": "brute-force angular search",
    "angles_per_frame_deg": [float(a) for a in angles_arr],
    "median_step_deg": float(np.median(np.abs(diffs))),
    "total_rotation_deg": float(abs(angles_arr[-1] - angles_arr[0])),
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose, f, indent=2)
print(f"\nSaved phase_angles.json")
