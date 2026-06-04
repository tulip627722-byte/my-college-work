"""Separate pot vs plant by color, fit rotation axis from pot only"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CROP = os.path.join(DATA, "output", "cropped_frames")
OUT = os.path.join(DATA, "output")

# Load frame 0 with colors
pcd = o3d.io.read_point_cloud(os.path.join(CROP, "frame_000_crop.ply"))
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)  # 0-1 range RGB

# Separate by greenness: ratio = G / (R+G+B)
r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
# Green plant: G is dominant channel
is_green = (g > r * 1.2) & (g > b * 1.1)
is_pot = ~is_green

pts_plant = pts[is_green]
clr_plant = colors[is_green]
pts_pot = pts[is_pot]
clr_pot = colors[is_pot]

print(f"Frame 0 separation:")
print(f"  Pot:   {len(pts_pot):5d} pts  X=[{pts_pot[:,0].min():.3f},{pts_pot[:,0].max():.3f}]  Y=[{pts_pot[:,1].min():.3f},{pts_pot[:,1].max():.3f}]")
print(f"  Plant: {len(pts_plant):5d} pts  X=[{pts_plant[:,0].min():.3f},{pts_plant[:,0].max():.3f}]  Y=[{pts_plant[:,1].min():.3f},{pts_plant[:,1].max():.3f}]")

# Save separated versions for visualization
pcd_pot = o3d.geometry.PointCloud()
pcd_pot.points = o3d.utility.Vector3dVector(pts_pot)
pcd_pot.colors = o3d.utility.Vector3dVector(clr_pot)
o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_pot.ply"), pcd_pot)

pcd_plant = o3d.geometry.PointCloud()
pcd_plant.points = o3d.utility.Vector3dVector(pts_plant)
pcd_plant.colors = o3d.utility.Vector3dVector(clr_plant)
o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_plant.ply"), pcd_plant)

# Save combined with labels for viz
pcd_both = o3d.geometry.PointCloud()
pcd_both.points = o3d.utility.Vector3dVector(np.vstack([pts_pot, pts_plant]))
clr_viz = np.zeros((len(pts_pot)+len(pts_plant), 3))
clr_viz[:len(pts_pot)] = [0.3, 0.3, 0.8]     # Blue = pot
clr_viz[len(pts_pot):] = [0.3, 0.8, 0.3]     # Green = plant
pcd_both.colors = o3d.utility.Vector3dVector(clr_viz)
o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_separated.ply"), pcd_both)
print(f"\nSaved: debug_frame000_pot.ply (blue), debug_frame000_plant.ply (green), debug_frame000_separated.ply")

# ============================================================
# Now fit rotation axis from POT centroids only (all 36 frames)
# ============================================================
print("\n" + "=" * 60)
print("Fitting rotation axis from POT points only")
print("=" * 60)

pot_centroids = []

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CROP, f"frame_{fidx:03d}_crop.ply"))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Separate pot by green filter
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    is_pot = ~((g > r * 1.2) & (g > b * 1.1))
    pts_pot_f = pts[is_pot]

    if len(pts_pot_f) < 50:
        print(f"Frame {fidx:03d}: pot too small ({len(pts_pot_f)} pts), skip")
        continue

    cx = np.mean(pts_pot_f[:, 0])
    cz = np.mean(pts_pot_f[:, 2])
    pot_centroids.append((fidx, cx, cz, len(pts_pot_f)))
    print(f"Frame {fidx:03d}: pot centroid=({cx:.4f},{cz:.4f})  n={len(pts_pot_f)}")

# Fit circle to pot centroids in XZ plane
cxs = np.array([c[1] for c in pot_centroids])
czs = np.array([c[2] for c in pot_centroids])

print(f"\nPot centroid distribution:")
print(f"  CX: mean={cxs.mean():.4f}, std={cxs.std():.4f}, median={np.median(cxs):.4f}")
print(f"  CZ: mean={czs.mean():.4f}, std={czs.std():.4f}, median={np.median(czs):.4f}")

# Use median as rotation center
rot_cx = np.median(cxs)
rot_cz = np.median(czs)
print(f"\n  Rotation axis (pot centroids median): X={rot_cx:.4f}, Z={rot_cz:.4f}")

# Compute radius from centroids
dists = np.sqrt((cxs - rot_cx)**2 + (czs - rot_cz)**2)
print(f"  Distance from center: mean={dists.mean():.4f}m, std={dists.std():.4f}m")

# Save for next steps
rot_params = {
    "rotation_center_x": float(rot_cx),
    "rotation_center_z": float(rot_cz),
    "pot_centroid_dist_mean": float(dists.mean()),
    "pot_centroid_dist_std": float(dists.std()),
}
with open(os.path.join(OUT, "rotation_axis.json"), "w") as f:
    json.dump(rot_params, f, indent=2)

print(f"\nSaved rotation_axis.json")
print(f"\nVerification items:")
print(f"  1. Open debug_frame000_separated.ply — check blue=pot, green=plant")
print(f"  2. Verify pot centroid std is small (< 0.02m ideal)")
