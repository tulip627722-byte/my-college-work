"""Verify 10 deg/frame by registering distant frame pairs, then assemble"""
import numpy as np
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
CLEAN = os.path.join(DATA, "output", "clean_frames")
OUT = os.path.join(DATA, "output")

rot_cx, rot_cz = -0.0346, 0.7861

# Load centered point clouds
pcds = []
for fidx in range(36):
    pcd = o3d.io.read_point_cloud(os.path.join(CLEAN, f"frame_{fidx:03d}_clean.ply"))
    pts = np.asarray(pcd.points)
    pts[:, 0] -= rot_cx
    pts[:, 2] -= rot_cz
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcds.append(pcd)

# Register distant pairs to estimate per-frame angle
pairs = [(0, 5), (0, 10), (0, 18), (0, 27), (5, 23), (10, 28)]
print("Distant-pair ICP to verify angle/frame:")
estimates = []
for i, j in pairs:
    reg = o3d.pipelines.registration.registration_icp(
        pcds[j], pcds[i], 0.03, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    R = reg.transformation[:3, :3]
    angle = abs(np.degrees(np.arctan2(R[0, 2], R[0, 0])))
    per_frame = angle / (j - i)
    estimates.append(per_frame)
    print(f"  Frame {i}->{j} (delta={j-i}): angle={angle:.1f} deg, per_frame={per_frame:.2f} deg, fitness={reg.fitness:.3f}")

print(f"\n  Estimated deg/frame: median={np.median(estimates):.2f}, mean={np.mean(estimates):.2f}")
print(f"  User says ~10 deg/frame -> using 10.0 deg/frame for assembly")
