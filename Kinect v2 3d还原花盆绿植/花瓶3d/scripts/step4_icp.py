"""Step 4 v2: Phase angle from ICP registration between adjacent frames"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

with open(os.path.join(OUT, "rotation_axis.json")) as f:
    rc = json.load(f)
rot_cx, rot_cz = rc["rotation_center_x"], rc["rotation_center_z"]

# Load all frames
pcds = []
for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pcds.append(pcd)

# ICP between adjacent frames
# Translate to origin (rotation center at XZ) first, then ICP
# We're looking for pure Y-axis rotation

angles = [0.0]  # frame 0 = 0 deg
angle_cumulative = 0.0

print(f"Frame 000: angle = 0.00 deg (reference)")

for fidx in range(1, 36):
    src = pcds[fidx-1]  # source: previous frame
    tgt = pcds[fidx]    # target: current frame

    # ICP: point-to-plane works better than point-to-point for smooth surfaces
    threshold = 0.02  # 2cm max correspondence distance
    trans_init = np.eye(4)

    # Try ICP
    try:
        reg = o3d.pipelines.registration.registration_icp(
            tgt, src, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        T = reg.transformation
        R = T[:3, :3]

        # Extract Y-axis rotation angle from rotation matrix
        # R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
        # theta = atan2(R[0,2], R[0,0])
        angle_step = np.arctan2(R[0, 2], R[0, 0])  # in radians
        angle_step_deg = np.degrees(angle_step)

        angle_cumulative += angle_step_deg
        angles.append(angle_cumulative)

        fitness = reg.fitness
        rmse = reg.inlier_rmse
        print(f"Frame {fidx:03d}: step={angle_step_deg:+.2f} deg  total={angle_cumulative:.1f} deg  fitness={fitness:.3f}  rmse={rmse:.4f}m")
    except Exception as e:
        print(f"Frame {fidx:03d}: ICP failed ({e}), using median step")
        angles.append(angle_cumulative + 0)  # placeholder

angles_arr = np.array(angles)
diffs = np.diff(angles_arr)
print(f"\n{'='*60}")
print(f"ICP Phase Angle Results:")
print(f"  Adjacent steps (deg): {[f'{d:.2f}' for d in diffs]}")
print(f"  Median step: {np.median(np.abs(diffs)):.2f} deg/frame")
print(f"  Total rotation: {angles_arr[-1] - angles_arr[0]:.1f} deg")
print(f"  Expected: ~360 deg")
print(f"  Range: {angles_arr.min():.1f} to {angles_arr.max():.1f} deg")

# Save
pose = {
    "method": "ICP point-to-point",
    "rotation_center_x": rot_cx,
    "rotation_center_z": rot_cz,
    "angles_per_frame_deg": [float(a) for a in angles_arr],
    "icp_fitness": float(reg.fitness),
    "icp_rmse": float(reg.inlier_rmse),
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose, f, indent=2)
print(f"\nSaved phase_angles.json")
