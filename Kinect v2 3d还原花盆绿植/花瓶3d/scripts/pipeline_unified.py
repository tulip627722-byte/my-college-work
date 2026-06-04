"""
Unified pipeline: per-frame pot+plant use SAME rot_center transform
Pot: 0.08 < Y < 0.20,  Plant: Y < 0.08
"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

FX,FY=365.456,365.456; CX,CY=254.878,205.395
ROT_CX,ROT_CZ=-0.0346,0.7861; DEG=360.0/36; N=36

def load_depth(i):
    raw=np.fromfile(os.path.join(DATA,"depth",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(raw,cv2.IMREAD_UNCHANGED)

def load_color_bgr(i):
    raw=np.fromfile(os.path.join(DATA,"color",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(raw,cv2.IMREAD_COLOR)

def rot_center(deg):
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    T1=np.eye(4); T1[0,3]=-ROT_CX; T1[2,3]=-ROT_CZ
    R=np.eye(4); R[0,0]=c; R[0,2]=s; R[2,0]=-s; R[2,2]=c
    T2=np.eye(4); T2[0,3]=ROT_CX; T2[2,3]=ROT_CZ
    return T2@R@T1

# Load spatial mask
mask_raw=np.fromfile(os.path.join(DATA,"output","mask_2d.png"),dtype=np.uint8)
mask_2d=cv2.imdecode(mask_raw,cv2.IMREAD_GRAYSCALE)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
spatial_mask=cv2.dilate(mask_2d,kernel,iterations=1)

# ============================================================
# Step 1: Per-frame extract pot + plant, apply SAME transform
# ============================================================
print("=" * 60)
print("Step 1: Per-frame extraction + transform")
print("=" * 60)

all_pts, all_clr = [], []

for i in range(N):
    depth = load_depth(i)
    depth_f = cv2.bilateralFilter(depth.astype(np.float32), 5, 30, 30) / 1000.0
    color_bgr = load_color_bgr(i)
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 3D points (camera coords)
    vv,uu = np.mgrid[0:424,0:512]; zz = depth_f; ok = zz > 0.01
    zv,uv,vv = zz[ok], uu[ok], vv[ok]
    x = (uv - CX) * zv / FX
    y = (vv - CY) * zv / FY
    pts = np.stack([x, y, zv], axis=1)  # (N,3) in camera coords

    # RGB lookup
    ur = np.clip((uv * 1920 // 512).astype(int), 0, 1919)
    vr = np.clip((vv * 1080 // 424).astype(int), 0, 1079)
    hsv = color_hsv[vr, ur]; rgb = color_rgb[vr, ur]
    h,s,v = hsv[:,0].astype(float), hsv[:,1].astype(float), hsv[:,2].astype(float)
    r,g,b = rgb[:,0].astype(float), rgb[:,1].astype(float), rgb[:,2].astype(float)
    in_mask = spatial_mask[vr, ur] > 0

    # Segmentation
    is_green = (h >= 35) & (h <= 85) & (s > 30) & (v > 30) & (g > r * 0.95)
    is_pot = (in_mask & ~is_green &
              (pts[:,1] > 0.08) & (pts[:,1] < 0.20) &
              (pts[:,2] > 0.4) & (pts[:,2] < 2.0))
    is_plant = (in_mask & is_green &
                (pts[:,1] < 0.08) & (pts[:,1] > -0.15) &
                (pts[:,2] > 0.3) & (pts[:,2] < 2.5))

    # Combined: pot + plant in camera coords
    keep = is_pot | is_plant
    frame_pts = pts[keep]
    frame_rgb = rgb[keep].astype(np.float64) / 255.0

    # SAME transform for both!
    T = rot_center(-i * DEG)
    frame_h = np.hstack([frame_pts, np.ones((len(frame_pts), 1))])
    world_pts = (T @ frame_h.T).T[:, :3]

    all_pts.append(world_pts)
    all_clr.append(frame_rgb)

    if i % 6 == 0:
        n_pot, n_plant = np.sum(is_pot), np.sum(is_plant)
        print(f"  Frame {i:03d}: pot={n_pot} plant={n_plant} -> world")

# ============================================================
# Step 2: Merge + cleanup
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Merge + cleanup")
print("=" * 60)

merged_pts = np.vstack(all_pts)
merged_clr = np.vstack(all_clr)
print(f"  Merged raw: {len(merged_pts)} pts")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_pts)
pcd.colors = o3d.utility.Vector3dVector(merged_clr)

pcd = pcd.voxel_down_sample(0.003)
print(f"  After voxel(3mm): {len(pcd.points)} pts")

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"  After SOR: {len(pcd.points)} pts")

# ============================================================
# Step 3: Save + outputs
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Save outputs")
print("=" * 60)

# PLY
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.ply"), pcd)
print(f"Saved: merged_cloud.ply ({len(pcd.points)} pts)")

# PCD (ASCII for Chinese path compat)
o3d.io.write_point_cloud(os.path.join(OUT, "merged_cloud.pcd"), pcd, write_ascii=True)
print(f"Saved: merged_cloud.pcd")

# pose.json
poses = {}
for i in range(N):
    poses[f"frame_{i:03d}"] = {
        "angle_deg": round(i * DEG, 1),
        "transform_world_from_camera": rot_center(-i * DEG).tolist()
    }
with open(os.path.join(OUT, "pose.json"), "w") as f:
    json.dump(poses, f, indent=2)
print("Saved: pose.json")

# report.json
report = {
    "pipeline": "unified_rot_center",
    "date": "2026-06-01",
    "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
    "deg_per_frame": DEG,
    "num_frames": N,
    "segmentation": {
        "pot": "0.08 < Y < 0.20, HSV low-sat, not green",
        "plant": "Y < 0.08, HSV green, RGB g>r*0.95"
    },
    "transform": "same rot_center(-i*10deg) for both pot and plant per frame",
    "total_points": len(pcd.points),
    "outputs": ["merged_cloud.ply", "merged_cloud.pcd", "pose.json", "report.json"]
}
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved: report.json")

# ============================================================
# Visualize
# ============================================================
print("\n" + "=" * 60)
print("FINAL MERGED CLOUD — close window to finish.")
print("=" * 60)
try:
    o3d.visualization.draw_geometries(
        [pcd], window_name="Final — Pot + Plant (unified transform)",
        width=1280, height=720)
except Exception as e:
    print(f"Viz: {e}")

print("\nAll done!")
