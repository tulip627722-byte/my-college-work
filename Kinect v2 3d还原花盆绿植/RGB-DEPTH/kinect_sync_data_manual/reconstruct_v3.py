"""
Kinect v2 转台三维重建  v3.0
============================
核心修复（vs v2.x）：
  - 正确的 RGB-D 对齐：彩色图(1920x1080) 用彩色相机内参采样，
    深度图(512x424) 用深度相机内参反投影，通过外参 R/T 正确映射
  - Y 轴自动翻转：输出模型正立（植物在上，花盆在下）
  - 背景自动剔除：只保留转台上的物体，去掉桌面/底座
  - 完整流程：反投影 → 粗对齐 → ICP精对齐 → 位姿图优化 → Poisson重建

用法
----
  pip install open3d numpy scipy Pillow
  python reconstruct_v3.py
  (脚本和数据放同一目录，或修改下方 DATA_DIR)
"""

import os, sys, glob
import numpy as np
import open3d as o3d
from scipy.ndimage import median_filter
from PIL import Image

# ═══════════════════════════════════════════════════════════════
#  Kinect v2 标准出厂内参
# ═══════════════════════════════════════════════════════════════

# 深度相机内参 (512x424)
D_FX, D_FY = 365.456, 365.456
D_CX, D_CY = 254.878, 205.395
DEPTH_W, DEPTH_H = 512, 424
DEPTH_SCALE = 0.001   # mm -> m

# 彩色相机内参 (1920x1080)
C_FX, C_FY = 1081.370, 1081.370
C_CX, C_CY = 959.500,  539.500
COLOR_W, COLOR_H = 1920, 1080

# 深度->彩色 外参（Kinect v2 标准值，单位 m）
# 彩色相机相对深度相机的平移（彩色在深度右侧约 52mm）
R_D2C = np.eye(3)   # 旋转极小，近似单位矩阵
T_D2C = np.array([-0.0520, 0.0, 0.0])   # x 方向平移 -52mm

# ═══════════════════════════════════════════════════════════════
#  数据集 & 处理参数
# ═══════════════════════════════════════════════════════════════

TOTAL_VIEWS  = 25
ANGLE_STEP   = 360.0 / TOTAL_VIEWS

# ROI（深度相机坐标系，米）
# Y轴：Kinect向下为正，花盆底约 y=0.22，植物顶约 y=-0.10
MAIN_ROI = dict(x=(-0.22, 0.22), y=(-0.15, 0.28), z=(0.40, 0.85))
BASE_ROI = dict(x=(-0.18, 0.18), y=( 0.03, 0.22), z=(0.45, 0.72))

# 处理参数
VOXEL_ICP          = 0.004
VOXEL_DENSE        = 0.003
VOXEL_CLEAN        = 0.005
CICP_DIST_COARSE   = 0.06
CICP_DIST_FINE     = 0.025
POISSON_DEPTH      = 9
POISSON_TRIM       = 0.06

# ═══════════════════════════════════════════════════════════════
#  RGB-D 对齐（核心修复）
# ═══════════════════════════════════════════════════════════════

def load_rgbd_frame(data_dir, frame_no):
    """
    正确的 RGB-D 对齐流程：
    1. 深度图反投影到 3D（深度相机坐标系）
    2. 通过外参 R/T 变换到彩色相机坐标系
    3. 用彩色相机内参投影到彩色图像平面，采样颜色
    """
    cp = os.path.join(data_dir, f"{frame_no:04d}_color.tif")
    dp = os.path.join(data_dir, f"{frame_no:04d}_depth.tif")
    if not os.path.isfile(cp): raise FileNotFoundError(cp)
    if not os.path.isfile(dp): raise FileNotFoundError(dp)

    # 读深度图
    depth = np.array(Image.open(dp)).astype(np.float32)
    if depth.shape != (DEPTH_H, DEPTH_W):
        raise ValueError(f"深度图尺寸错误: {depth.shape}, 期望({DEPTH_H},{DEPTH_W})")
    depth = median_filter(depth, size=3).astype(np.float32)

    # 读彩色图（保持原始 1920x1080）
    color = np.array(Image.open(cp).convert("RGB"), dtype=np.uint8)
    if color.shape[:2] != (COLOR_H, COLOR_W):
        color = np.array(
            Image.open(cp).convert("RGB").resize((COLOR_W, COLOR_H), Image.BILINEAR),
            dtype=np.uint8)

    # ── Step1: 深度反投影到深度相机 3D 坐标 ──────────────────────────
    u, v = np.meshgrid(np.arange(DEPTH_W), np.arange(DEPTH_H))
    Z_d = depth * DEPTH_SCALE                    # (H, W)
    X_d = (u - D_CX) * Z_d / D_FX
    Y_d = (v - D_CY) * Z_d / D_FY

    # 有效深度掩码
    valid = (Z_d > 0.1) & (Z_d < 5.0)
    pts_d = np.stack([X_d[valid], Y_d[valid], Z_d[valid]], axis=-1)  # (N, 3)

    # ── Step2: 深度相机坐标 → 彩色相机坐标 ──────────────────────────
    pts_c = (R_D2C @ pts_d.T).T + T_D2C          # (N, 3)

    # ── Step3: 彩色相机坐标 → 彩色图像像素 ──────────────────────────
    # 避免除以零
    Z_c = pts_c[:, 2]
    valid2 = Z_c > 0.01
    pts_c = pts_c[valid2]
    Z_c = Z_c[valid2]

    u_c = (C_FX * pts_c[:, 0] / Z_c + C_CX).astype(np.float32)
    v_c = (C_FY * pts_c[:, 1] / Z_c + C_CY).astype(np.float32)

    # 边界检查
    in_bounds = ((u_c >= 0) & (u_c < COLOR_W - 1) &
                 (v_c >= 0) & (v_c < COLOR_H - 1))

    pts_final = pts_c[in_bounds]   # 深度相机坐标系的3D点
    u_f = u_c[in_bounds].astype(int)
    v_f = v_c[in_bounds].astype(int)

    # 双线性插值采样颜色（用整数近似）
    cols_final = color[v_f, u_f]   # (N, 3) uint8

    return pts_final, cols_final


def extract_roi(pts, cols, roi):
    mask = ((pts[:,0] >= roi['x'][0]) & (pts[:,0] <= roi['x'][1]) &
            (pts[:,1] >= roi['y'][0]) & (pts[:,1] <= roi['y'][1]) &
            (pts[:,2] >= roi['z'][0]) & (pts[:,2] <= roi['z'][1]))
    return pts[mask], cols[mask]


def make_cloud(pts, cols):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    return pc


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def rotate_pts(pts, center, theta):
    return (rot_y(theta) @ (pts - center).T).T + center


def ensure_normals(pc, radius=0.015, max_nn=30):
    if not pc.has_normals():
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pc.orient_normals_consistent_tangent_plane(30)
    return pc


# ═══════════════════════════════════════════════════════════════
#  旋转中心估算 + 手动选点
# ═══════════════════════════════════════════════════════════════

def estimate_center_auto(base_pts_list):
    pts = np.vstack(base_pts_list)
    x, z = pts[:,0], pts[:,2]
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    x, z = x[(x>=lo)&(x<=hi)], z[(x>=lo)&(x<=hi)]
    A = np.column_stack([2*x, 2*z, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x**2+z**2, rcond=None)
    return np.array([sol[0], 0.0, sol[1]])


def pick_center_interactive(ref_cloud):
    print("\n" + "="*55)
    print("手动选转台旋转中心")
    print("  1. 将视角调整为正上方俯视")
    print("  2. Shift + 左键点击花盆底座圆心")
    print("  3. 可多次点击，关闭窗口继续")
    print("="*55)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Shift+Click 选转台圆心 → 关闭窗口", 1024, 768)
    vis.add_geometry(ref_cloud)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    pts = np.asarray(ref_cloud.points)
    if len(idx) == 0:
        print("  未选点，使用自动估算")
        return None
    sel = pts[list(idx)]
    cx, cz = float(np.mean(sel[:,0])), float(np.mean(sel[:,2]))
    print(f"  → 中心: X={cx:.4f}  Z={cz:.4f}")
    return np.array([cx, 0.0, cz])


# ═══════════════════════════════════════════════════════════════
#  ICP + 位姿图优化
# ═══════════════════════════════════════════════════════════════

def pairwise_icp(src, tgt, voxel, dist_coarse, dist_fine):
    sd = src.voxel_down_sample(voxel)
    td = tgt.voxel_down_sample(voxel)
    ensure_normals(sd); ensure_normals(td)
    r1 = o3d.pipelines.registration.registration_icp(
        sd, td, dist_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    r2 = o3d.pipelines.registration.registration_colored_icp(
        sd, td, dist_fine, r1.transformation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50))
    return r2.transformation, r2.inlier_rmse


def build_pose_graph(clouds, voxel, dist_coarse, dist_fine):
    n = len(clouds)
    pg = o3d.pipelines.registration.PoseGraph()
    odom = np.eye(4)
    pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))
    print("  相邻帧 ICP…")
    for i in range(n-1):
        T, rmse = pairwise_icp(clouds[i+1], clouds[i],
                               voxel, dist_coarse, dist_fine)
        odom = T @ odom
        pg.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            clouds[i+1].voxel_down_sample(voxel),
            clouds[i  ].voxel_down_sample(voxel), dist_fine, T)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i+1, i, T, info, uncertain=False))
        print(f"    帧{i+1:02d}→{i:02d}  RMSE={rmse:.5f}")
    print("  闭环约束…")
    T_loop, rmse_loop = pairwise_icp(clouds[n-1], clouds[0],
                                     voxel, dist_coarse, dist_fine)
    info_loop = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        clouds[n-1].voxel_down_sample(voxel),
        clouds[0  ].voxel_down_sample(voxel), dist_fine, T_loop)
    pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
        n-1, 0, T_loop, info_loop, uncertain=True))
    print(f"    闭环 RMSE={rmse_loop:.5f}")
    print("  全局优化…")
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=dist_fine,
            edge_prune_threshold=0.25, reference_node=0))
    return [pg.nodes[i].pose for i in range(n)]


# ═══════════════════════════════════════════════════════════════
#  颜色修复
# ═══════════════════════════════════════════════════════════════

def repair_colors(pc):
    from colorsys import rgb_to_hsv
    pts  = np.asarray(pc.points)
    cols = np.asarray(pc.colors)
    hsv  = np.array([rgb_to_hsv(*c) for c in cols])
    H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]
    # 翻转后 Y>0 是植物区域
    plant = pts[:,1] > -0.05
    green = plant & (H>=0.08)&(H<=0.45)&(S>=0.18)&(V>=0.08)
    gray  = plant & (S<=0.22)&(V>=0.35)
    if green.sum() < 20 or gray.sum() == 0:
        return pc
    g_cloud = o3d.geometry.PointCloud()
    g_cloud.points = o3d.utility.Vector3dVector(pts[green])
    kd = o3d.geometry.KDTreeFlann(g_cloud)
    cols2 = cols.copy()
    recolored = 0
    for gi in np.where(gray)[0]:
        k, idx, d2 = kd.search_knn_vector_3d(pts[gi], 5)
        if k == 0 or np.sqrt(d2[0]) > 0.02:
            continue
        cols2[gi] = cols[green][idx[:k]].mean(0)
        recolored += 1
    print(f"  颜色修复: 绿={green.sum()} 灰白={gray.sum()} 修复={recolored}")
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(np.clip(cols2, 0, 1))
    return out


# ═══════════════════════════════════════════════════════════════
#  Poisson 重建
# ═══════════════════════════════════════════════════════════════

def poisson_mesh(pc):
    pc2 = ensure_normals(pc.voxel_down_sample(VOXEL_CLEAN))
    print(f"  Poisson depth={POISSON_DEPTH}, 点数={len(pc2.points)}…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc2, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False)
    dens = np.asarray(densities)
    mesh.remove_vertices_by_mask(dens <= np.quantile(dens, POISSON_TRIM))
    # 颜色插值
    mpts = np.asarray(mesh.vertices)
    pcpts = np.asarray(pc2.points)
    pccols = np.asarray(pc2.colors)
    kd = o3d.geometry.KDTreeFlann(pc2)
    vcols = np.zeros((len(mpts), 3))
    for i, v in enumerate(mpts):
        _, idx, _ = kd.search_knn_vector_3d(v, 1)
        vcols[i] = pccols[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    mesh.compute_vertex_normals()
    print(f"  网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")
    return mesh


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(data_dir, "0001_color.tif")):
        data_dir = os.getcwd()
    print(f"数据目录: {data_dir}")

    # ── 1. 读取所有帧（正确 RGB-D 对齐）──────────────────────────────
    print("\n" + "="*55)
    print(f"读取 {TOTAL_VIEWS} 帧（正确 RGB-D 对齐）…")
    print("="*55)
    all_pts, all_cols, base_pts = [], [], []

    for i in range(1, TOTAL_VIEWS+1):
        pts, cols = load_rgbd_frame(data_dir, i)
        mp, mc = extract_roi(pts, cols, MAIN_ROI)
        bp, _  = extract_roi(pts, cols, BASE_ROI)
        all_pts.append(mp)
        all_cols.append(mc)
        base_pts.append(bp if len(bp) > 0 else mp)
        print(f"  帧{i:02d}: 总={len(pts):6,}  ROI内={len(mp):5,}  底座={len(bp):4,}")

    # ── 2. 估算旋转中心 ──────────────────────────────────────────────
    print("\n" + "="*55)
    auto_c = estimate_center_auto(base_pts)
    print(f"自动估算中心: X={auto_c[0]:.4f}  Z={auto_c[2]:.4f}")

    ans = input("\n打开交互选点窗口？[Y/n]: ").strip().lower()
    if ans != 'n':
        ref = make_cloud(all_pts[0], all_cols[0])
        picked = pick_center_interactive(ref)
        center = picked if picked is not None else auto_c
    else:
        center = auto_c
        print(f"使用自动中心: {center}")

    # ── 3. 两个方向都跑，选最好的 ───────────────────────────────────
    for rot_sign in [1, -1]:
        sign_name = "pos" if rot_sign > 0 else "neg"
        print(f"\n{'='*55}")
        print(f"方向 {rot_sign:+d} ({sign_name})")
        print("="*55)

        # 粗对齐
        clouds = []
        for i in range(TOTAL_VIEWS):
            theta = np.deg2rad(rot_sign * i * ANGLE_STEP)
            pts_r = rotate_pts(all_pts[i], center, theta)
            clouds.append(make_cloud(pts_r, all_cols[i]))

        # 精对齐
        print("\n位姿图精对齐…")
        poses = build_pose_graph(clouds, VOXEL_ICP,
                                 CICP_DIST_COARSE, CICP_DIST_FINE)

        # 合并
        print("\n合并…")
        merged = o3d.geometry.PointCloud()
        for cloud, pose in zip(clouds, poses):
            merged += cloud.transform(pose)
        print(f"  合并点数: {len(merged.points):,}")

        # ── Y轴翻转：Kinect Y向下 → 翻转后正立 ──────────────────────
        pts_arr = np.asarray(merged.points).copy()
        pts_arr[:, 1] *= -1
        merged.points = o3d.utility.Vector3dVector(pts_arr)
        print("  Y轴已翻转（模型正立）")

        # 去噪 + 颜色修复
        pc_clean = merged.voxel_down_sample(VOXEL_CLEAN)
        pc_clean, _ = pc_clean.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        pc_clean = repair_colors(pc_clean)

        pc_dense = merged.voxel_down_sample(VOXEL_DENSE)
        pc_dense = repair_colors(pc_dense)

        # 保存点云
        cloud_path = os.path.join(data_dir, f"v3_cloud_{sign_name}.ply")
        o3d.io.write_point_cloud(cloud_path, pc_clean)
        print(f"  点云 → {cloud_path}  ({len(pc_clean.points):,} pts)")

        # Poisson
        print("\nPoisson 重建…")
        mesh = poisson_mesh(pc_dense)
        mesh_path = os.path.join(data_dir, f"v3_mesh_{sign_name}.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=False)
        print(f"  网格 → {mesh_path}")

    print("\n" + "="*55)
    print("完成！")
    print("  MeshLab 打开 v3_mesh_pos.ply 或 v3_mesh_neg.ply")
    print("  显示颜色：Render > Color > Per Vertex")
    print("  哪个更完整就用哪个")
    print("="*55)


if __name__ == "__main__":
    main()
