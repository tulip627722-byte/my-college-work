"""Step 4 v3: Register each frame directly to frame 0"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

# Load frames
pcds = []
for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    # Move to origin (around rotation center)
    pts = np.asarray(pcd.points)
    pts[:, 0] -= -0.0346  # recenter X
    pts[:, 2] -= 0.7861   # recenter Z
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcds.append(pcd)

ref = pcds[0]  # reference = frame 0
angles_abs = [0.0]

print("Direct registration to frame 0:")
print(f"Frame 000: 0.0 deg")

for fidx in range(1, 36):
    tgt = pcds[fidx]
    threshold = 0.03

    try:
        reg = o3d.pipelines.registration.registration_icp(
            tgt, ref, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        R = reg.transformation[:3, :3]
        angle = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
        # Normalize to a consistent direction
        angles_abs.append(angle)
        print(f"Frame {fidx:03d}: {angle:+.1f} deg  fitness={reg.fitness:.3f}")
    except Exception as e:
        print(f"Frame {fidx:03d}: FAILED ({e})")
        angles_abs.append(np.nan)

# Report
arr = np.array(angles_abs)
valid = ~np.isnan(arr)
print(f"\nDirect ICP to frame 0:")
print(f"  Min: {arr[valid].min():.1f}  Max: {arr[valid].max():.1f}")
print(f"  Half-turn frame (closest to 180 deg): frame {np.argmin(np.abs(arr + 180))}  angle={arr[np.argmin(np.abs(arr + 180))]:.1f} deg")

# Also compute pairwise (frame to frame) for comparison
# Check if frame 0 and frame 18 align as 180 degrees apart
mid = 18
reg_mid = o3d.pipelines.registration.registration_icp(
    pcds[mid], ref, 0.03, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
angle_mid = np.degrees(np.arctan2(reg_mid.transformation[0, 2], reg_mid.transformation[0, 0]))
print(f"  Frame 0 -> Frame {mid}: {angle_mid:.1f} deg (expected ~180 if 360 total)")
print(f"  Implies total rotation: {abs(angle_mid) * 2:.0f} deg")

# Also try frame 35
reg_last = o3d.pipelines.registration.registration_icp(
    pcds[35], ref, 0.03, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
angle_last = np.degrees(np.arctan2(reg_last.transformation[0, 2], reg_last.transformation[0, 0]))
print(f"  Frame 0 -> Frame 35: {angle_last:.1f} deg (should be ~-10 if 360 total)")

# Save
pose = {
    "method": "ICP direct to frame 0",
    "angles_per_frame_deg": [float(a) for a in angles_abs],
    "mid_frame_angle_deg": float(angle_mid),
    "last_frame_angle_deg": float(angle_last),
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose, f, indent=2)
print(f"\nSaved phase_angles.json")
