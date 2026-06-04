"""
Pipeline v2 — 4-stage TSDF reconstruction
Stage1: Preprocess (bilateral + color seg per frame)
Stage2: Pot registration (10° theory init → p2plane ICP → pose graph → TSDF)
Stage3: Plant registration (pot optimized poses as init → p2plane ICP → TSDF)
Stage4: Fusion
"""
import numpy as np
import cv2
import open3d as o3d
import os, json, time

# ============================================================
# Config
# ============================================================
DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

FX, FY = 365.456, 365.456
CX, CY = 254.878, 205.395
ROT_CX, ROT_CZ = -0.0346, 0.7861
DEG = 360.0 / 36
N = 36

INTRINSIC = o3d.camera.PinholeCameraIntrinsic(512, 424, FX, FY, CX, CY)
VOXEL_LEN = 0.002
SDF_TRUNC = VOXEL_LEN * 5

# ============================================================
# Helpers
# ============================================================
def load_depth(i):
    raw = np.fromfile(os.path.join(DATA, "depth", f"{i:03d}.png"), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

def load_color_bgr(i):
    raw = np.fromfile(os.path.join(DATA, "color", f"{i:03d}.png"), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

def rot_around_center(angle_deg):
    """4x4 rotation around Y at (ROT_CX, 0, ROT_CZ)"""
    a = np.radians(angle_deg); c, s = np.cos(a), np.sin(a)
    t1 = np.eye(4); t1[0,3] = -ROT_CX; t1[2,3] = -ROT_CZ
    r = np.eye(4); r[0,0]=c; r[0,2]=s; r[2,0]=-s; r[2,2]=c
    t2 = np.eye(4); t2[0,3] = ROT_CX; t2[2,3] = ROT_CZ
    return t2 @ r @ t1

def cam_pose(i):
    """Camera pose in world frame (rotation center = origin). Camera rotates around object."""
    a = np.radians(-i * DEG); c, s = np.cos(a), np.sin(a)
    eye = np.array([ROT_CX*c - ROT_CZ*s, 0.0, ROT_CX*s + ROT_CZ*c])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, -1.0, 0.0])
    z = center - eye; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3,0]=x; T[:3,1]=y; T[:3,2]=z; T[:3,3]=eye
    return T

# ============================================================
# STAGE 1: Preprocessing
# ============================================================
print("=" * 60)
print("STAGE 1: Per-frame preprocessing")
print("=" * 60)

mask_raw = np.fromfile(os.path.join(DATA, "output", "mask_2d.png"), dtype=np.uint8)
mask_2d = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
spatial_mask = cv2.dilate(mask_2d, kernel, iterations=1)

pot_pcds, plant_pcds = [], []
pot_normals, plant_normals = [], []

for i in range(N):
    depth = load_depth(i)
    depth_m = depth.astype(np.float64) / 1000.0
    color = cv2.cvtColor(load_color_bgr(i), cv2.COLOR_BGR2RGB)

    # Bilateral filter depth
    depth_f = cv2.bilateralFilter(depth.astype(np.float32), 5, 30, 30) / 1000.0

    # Generate all points
    vv, uu = np.mgrid[0:424, 0:512]
    zz = depth_f
    ok = zz > 0.01
    zv, uv, vv = zz[ok], uu[ok], vv[ok]
    x = (uv - CX) * zv / FX
    y = (vv - CY) * zv / FY
    pts = np.stack([x, y, zv], axis=1)

    # Color lookup
    ur = np.clip((uv * 1920 // 512).astype(int), 0, 1919)
    vr = np.clip((vv * 1080 // 424).astype(int), 0, 1079)
    clr = color[vr, ur]

    in_mask = spatial_mask[vr, ur] > 0

    # Greenness
    r, g, b = clr[:,0].astype(float), clr[:,1].astype(float), clr[:,2].astype(float)
    green_ratio = g / (r + g + b + 0.001)
    is_green = (green_ratio > 0.37) & (g > r * 0.95)

    # --- Pot ---
    is_pot = (in_mask & (pts[:,1] > 0.06) & (pts[:,2] > 0.4) & (pts[:,2] < 2.0) &
              (~is_green | (pts[:,1] > 0.10)))
    pcd_pot = o3d.geometry.PointCloud()
    pcd_pot.points = o3d.utility.Vector3dVector(pts[is_pot])
    pcd_pot = pcd_pot.voxel_down_sample(0.003)
    pcd_pot, _ = pcd_pot.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_pot.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    pot_pcds.append(pcd_pot)

    # --- Plant ---
    is_plant = (in_mask & is_green & (pts[:,1] < 0.10) & (pts[:,1] > -0.15) &
                (pts[:,2] > 0.3) & (pts[:,2] < 2.5))
    pcd_plant = o3d.geometry.PointCloud()
    pcd_plant.points = o3d.utility.Vector3dVector(pts[is_plant])
    pcd_plant = pcd_plant.voxel_down_sample(0.003)
    pcd_plant, _ = pcd_plant.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.0)
    pcd_plant.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    plant_pcds.append(pcd_plant)

    print(f"Frame {i:03d}: pot={len(pcd_pot.points)} plant={len(pcd_plant.points)}")

# ============================================================
# STAGE 2: Pot registration + TSDF
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: Pot registration (point-to-plane ICP + pose graph)")
print("=" * 60)

pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
pot_poses = [np.eye(4)]

for i in range(1, N):
    src = pot_pcds[i]; tgt = pot_pcds[i-1]
    ns, nt = len(src.points), len(tgt.points)

    # Init: rotate src by -DEG (undo turntable rotation) to match tgt
    init = rot_around_center(-DEG)

    try:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.05, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50))
        T = reg.transformation
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src, tgt, 0.05, T)
        print(f"Frame {i:03d} pot ICP: fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}m")
    except Exception as e:
        print(f"Frame {i:03d} pot ICP failed ({e}), using nominal")
        T = init
        info = np.eye(6) * 0.01

    new_pose = pot_poses[-1] @ T
    pot_poses.append(new_pose)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(new_pose))
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(i-1, i, T, info, uncertain=False))

# Loop closure
print("Loop closure...")
init_loop = rot_around_center(-DEG)
try:
    reg_lc = o3d.pipelines.registration.registration_icp(
        pot_pcds[0], pot_pcds[N-1], 0.05, init_loop,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    info_lc = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pot_pcds[0], pot_pcds[N-1], 0.05, reg_lc.transformation)
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(N-1, 0, reg_lc.transformation, info_lc, uncertain=False))
    print(f"  fitness={reg_lc.fitness:.3f} rmse={reg_lc.inlier_rmse:.4f}m")
except Exception as e:
    print(f"  failed: {e}")

print("Optimizing pose graph...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=0.05, edge_prune_threshold=0.25, reference_node=0)
o3d.pipelines.registration.global_optimization(
    pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

opt_poses = [node.pose for node in pose_graph.nodes]
print(f"Pose graph: {len(opt_poses)} nodes optimized")

# --- TSDF Pot ---
print("\n" + "=" * 60)
print("STAGE 2b: TSDF pot mesh")
print("=" * 60)

tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_LEN, sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(N):
    depth = load_depth(i)
    depth_f = cv2.bilateralFilter(depth.astype(np.float32), 5, 30, 30)

    color_bgr = load_color_bgr(i)
    color_bgr_small = cv2.resize(color_bgr, (512, 424))

    depth_img = o3d.geometry.Image(depth_f.astype(np.uint16))
    color_img = o3d.geometry.Image(cv2.cvtColor(color_bgr_small, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False)

    tsdf.integrate(rgbd, INTRINSIC, np.linalg.inv(cam_pose(i)))
    if i % 6 == 0:
        print(f"  Frame {i:03d} integrated")

print("Extracting pot mesh...")
pot_mesh = tsdf.extract_triangle_mesh()
pot_mesh = pot_mesh.simplify_vertex_clustering(0.002, contraction=o3d.geometry.SimplificationContraction.Average)
pot_mesh = pot_mesh.filter_smooth_simple(3)
pot_mesh.remove_degenerate_triangles()
pot_mesh.remove_duplicated_vertices()
print(f"Pot mesh: {len(pot_mesh.vertices)} verts, {len(pot_mesh.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT, "pot_mesh.ply"), pot_mesh)

# ============================================================
# STAGE 3: Plant registration + TSDF
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: Plant registration (pot poses as init)")
print("=" * 60)

plant_poses = [np.eye(4)]
plant_poses_absolute = [np.eye(4)]

for i in range(1, N):
    src = plant_pcds[i]

    # Use pot's optimized pose difference as initial guess
    # pot_poses[i] = pot_poses[i-1] @ T_pot(i), so T_between = inv(pot_poses[i-1]) @ pot_poses[i]
    T_pot_init = np.linalg.inv(opt_poses[i-1]) @ opt_poses[i]

    if len(src.points) < 30:
        plant_poses.append(T_pot_init)
        plant_poses_absolute.append(plant_poses_absolute[-1] @ T_pot_init)
        print(f"Frame {i:03d}: too few plant pts, using pot pose")
        continue

    # Prepare target: previous frame plant points transformed to world
    tgt_pts = np.asarray(plant_pcds[i-1].points)
    tgt_n = np.asarray(plant_pcds[i-1].normals)
    tgt_pts_w = (plant_poses_absolute[-1][:3,:3] @ tgt_pts.T).T + plant_poses_absolute[-1][:3,3]
    # For normals: only rotate, no translation
    tgt_n_w = (plant_poses_absolute[-1][:3,:3] @ tgt_n.T).T
    tgt_n_w = tgt_n_w / (np.linalg.norm(tgt_n_w, axis=1, keepdims=True) + 1e-8)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts_w)
    tgt.normals = o3d.utility.Vector3dVector(tgt_n_w)

    # Multi-scale ICP
    T_current = np.eye(4)
    for threshold in [0.04, 0.02, 0.01]:
        try:
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, threshold, T_current,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T_current = reg.transformation
            print(f"  Frame {i:03d} scale({threshold}): fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}")
        except Exception:
            pass

    plant_poses.append(T_current)
    plant_poses_absolute.append(plant_poses_absolute[-1] @ T_current)

# --- TSDF Plant ---
print("\n" + "=" * 60)
print("STAGE 3b: TSDF plant mesh")
print("=" * 60)

tsdf_p = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_LEN, sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(N):
    depth = load_depth(i)
    depth_f = cv2.bilateralFilter(depth.astype(np.float32), 5, 30, 30)
    color_bgr = load_color_bgr(i)
    color_small = cv2.resize(color_bgr, (512, 424))

    depth_img = o3d.geometry.Image(depth_f.astype(np.uint16))
    color_img = o3d.geometry.Image(cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, depth_scale=1000.0, depth_trunc=3.0,
        convert_rgb_to_intensity=False)

    tsdf_p.integrate(rgbd, INTRINSIC, np.linalg.inv(cam_pose(i)))
    if i % 6 == 0:
        print(f"  Frame {i:03d} integrated")

print("Extracting plant mesh...")
plant_mesh = tsdf_p.extract_triangle_mesh()
plant_mesh = plant_mesh.simplify_vertex_clustering(0.002, contraction=o3d.geometry.SimplificationContraction.Average)
plant_mesh = plant_mesh.filter_smooth_simple(2)
plant_mesh.remove_degenerate_triangles()
plant_mesh.remove_duplicated_vertices()
print(f"Plant mesh: {len(plant_mesh.vertices)} verts, {len(plant_mesh.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT, "plant_mesh.ply"), plant_mesh)

# ============================================================
# STAGE 4: Fusion
# ============================================================
print("\n" + "=" * 60)
print("STAGE 4: Fusion")
print("=" * 60)

fused = pot_mesh + plant_mesh
fused.remove_duplicated_vertices()
fused.remove_degenerate_triangles()
fused = fused.filter_smooth_taubin(number_of_iterations=5)
print(f"Fused: {len(fused.vertices)} verts, {len(fused.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.ply"), fused)
o3d.io.write_triangle_mesh(os.path.join(OUT, "fused_mesh.obj"), fused)

# Metadata
info = {
    "pipeline": "v2_tsdf",
    "voxel_length_m": VOXEL_LEN,
    "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
    "deg_per_frame": DEG,
    "pot_mesh": {"verts": len(pot_mesh.vertices), "tris": len(pot_mesh.triangles)},
    "plant_mesh": {"verts": len(plant_mesh.vertices), "tris": len(plant_mesh.triangles)},
    "fused_mesh": {"verts": len(fused.vertices), "tris": len(fused.triangles)},
}
with open(os.path.join(OUT, "report_v2.json"), "w") as f:
    json.dump(info, f, indent=2)

print("\n" + "=" * 60)
print("DONE. Files: pot_mesh.ply plant_mesh.ply fused_mesh.{ply,obj}")
print("=" * 60)
