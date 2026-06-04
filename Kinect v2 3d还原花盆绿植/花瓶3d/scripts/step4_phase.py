"""Step 4: Phase angle estimation from plant centroid rotation"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

# Load rotation center
with open(os.path.join(OUT, "rotation_axis.json")) as f:
    rc = json.load(f)
rot_cx = rc["rotation_center_x"]
rot_cz = rc["rotation_center_z"]
print(f"Rotation center: ({rot_cx:.4f}, {rot_cz:.4f})")

Y_PLANT_HI = 0.08
angles = []

for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)

    # Plant only
    plant_pts = pts[pts[:, 1] < Y_PLANT_HI]
    if len(plant_pts) < 10:
        # Fallback: use pot + plant centroid for phase
        plant_pts = pts
        print(f"Frame {fidx:03d}: using all pts for angle (plant too small)")

    mx = np.mean(plant_pts[:, 0])
    mz = np.mean(plant_pts[:, 2])

    # Angle relative to rotation center
    dx = mx - rot_cx
    dz = mz - rot_cz
    angle = np.arctan2(dx, dz)  # radians, range [-pi, pi]
    angles.append(angle)

    dist = np.sqrt(dx*dx + dz*dz)
    print(f"Frame {fidx:03d}: plant centroid=({mx:.4f},{mz:.4f})  angle={np.degrees(angle):.1f} deg  dist={dist:.4f}m")

# Convert to continuous increasing angles (unwrap)
angles_unwrapped = np.unwrap(angles)
print(f"\nUnwrapped angles (rad): {angles_unwrapped}")
print(f"Unwrapped angles (deg): {np.degrees(angles_unwrapped)}")

# Check monotonicity
diffs = np.diff(angles_unwrapped)
print(f"\nAdjacent frame angle diffs (deg):")
for i, d in enumerate(np.degrees(diffs)):
    marker = "" if d > 0 else "  <<< NEGATIVE!"
    print(f"  {i:03d}->{i+1:03d}: {d:.2f} deg{marker}")

n_neg = np.sum(diffs < 0)
if n_neg > 0:
    print(f"\nWARNING: {n_neg}/{len(diffs)} frames have negative angle change!")

deg_per_frame = np.median(diffs[diffs > 0]) if n_neg < len(diffs) else np.abs(np.median(diffs))
total_deg = (len(diffs)) * deg_per_frame

print(f"\n{'='*60}")
print(f"Phase angle estimation results:")
print(f"  Median angle step: {np.degrees(deg_per_frame):.2f} deg/frame")
print(f"  Total rotation:    {np.degrees(total_deg):.1f} deg over {len(diffs)} steps")
print(f"  Expected:          ~360 deg (10 deg/frame for 36 frames)")
print(f"  Unwrapped range:   {np.degrees(angles_unwrapped[0]):.1f} to {np.degrees(angles_unwrapped[-1]):.1f} deg")

# Save angles for Step 5
pose_data = {
    "rotation_center_x": rot_cx,
    "rotation_center_z": rot_cz,
    "deg_per_frame": float(np.degrees(deg_per_frame)),
    "total_rotation_deg": float(np.degrees(total_deg)),
    "angles_per_frame_deg": [float(np.degrees(a)) for a in angles_unwrapped],
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose_data, f, indent=2)

print(f"\nSaved phase_angles.json")
