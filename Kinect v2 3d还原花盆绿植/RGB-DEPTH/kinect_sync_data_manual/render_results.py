"""
将 reconstruct_v2 的输出 (.ply) 渲染为多角度 PNG 截图  v2
策略：旋转网格本身而非移动相机，确保每帧都正确渲染
用法: python render_results.py
"""
import os
import numpy as np
import open3d as o3d

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rot_y(theta):
    """绕 Y 轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(theta):
    """绕 X 轴旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def render_mesh_angle(mesh, name, angle_name, size=(1400, 1000)):
    """渲染网格的单个角度，旋转网格后从正面拍摄"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{name}_{angle_name}",
                      width=size[0], height=size[1],
                      visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.mesh_show_back_face = True

    ctrl = vis.get_view_control()
    cam_params = ctrl.convert_to_pinhole_camera_parameters()
    # 使用默认相机的 intrinsics，只调整 extrinsics 为正面
    center = mesh.get_center()
    # 相机放在 object 正前上方
    cam_params.extrinsic = np.array([
        [1, 0, 0, -center[0]],
        [0, -0.866, -0.5, -center[1] + 0.25],
        [0, -0.5, 0.866, -center[2] + 0.35],
        [0, 0, 0, 1],
    ])
    ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    out_path = os.path.join(OUTPUT_DIR, f"{name}_{angle_name}.png")
    vis.capture_screen_image(out_path, do_render=True)
    print(f"  → {os.path.basename(out_path)}")
    vis.destroy_window()


def render_mesh(mesh_path, name, size=(1400, 1000)):
    """旋转网格，从多个角度渲染"""
    if not os.path.isfile(mesh_path):
        print(f"  跳过: {mesh_path}")
        return

    original = o3d.io.read_triangle_mesh(mesh_path)
    print(f"  渲染 {name}: {len(original.vertices)} 顶点, {len(original.triangles)} 面")

    if not original.has_vertex_normals():
        original.compute_vertex_normals()

    # 需要手动旋转的视角
    angles = {
        "front":  (0,      0),
        "right":  (0,      np.deg2rad(-90)),
        "back":   (0,      np.deg2rad(-180)),
        "left":   (0,      np.deg2rad(-270)),
        "top":    (np.deg2rad(90),  0),
        "persp":  (0,      np.deg2rad(-45)),
    }

    for label, (pitch, yaw) in angles.items():
        mesh_copy = o3d.geometry.TriangleMesh(original)
        verts = np.asarray(mesh_copy.vertices)
        center = mesh_copy.get_center()
        # 1. 平移到原点 → 2. 旋转 (先 Y 后 X) → 3. 平移回去
        verts = verts - center
        verts = (rot_y(yaw) @ verts.T).T
        verts = (rot_x(pitch) @ verts.T).T
        verts = verts + center
        mesh_copy.vertices = o3d.utility.Vector3dVector(verts)
        if mesh_copy.has_vertex_normals():
            norms = np.asarray(mesh_copy.vertex_normals)
            norms = (rot_y(yaw) @ norms.T).T
            norms = (rot_x(pitch) @ norms.T).T
            mesh_copy.vertex_normals = o3d.utility.Vector3dVector(norms)

        render_mesh_angle(mesh_copy, name, label, size)


def render_cloud(cloud_path, name, size=(1400, 1000)):
    """渲染点云"""
    if not os.path.isfile(cloud_path):
        print(f"  跳过: {cloud_path}")
        return

    cloud = o3d.io.read_point_cloud(cloud_path)
    print(f"  渲染 {name}: {len(cloud.points)} 点")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=size[0], height=size[1],
                      visible=False)
    vis.add_geometry(cloud)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 2.5

    ctrl = vis.get_view_control()
    center = cloud.get_center()
    cam_params = ctrl.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = np.array([
        [1, 0, 0, -center[0]],
        [0, -0.866, -0.5, -center[1] + 0.25],
        [0, -0.5, 0.866, -center[2] + 0.35],
        [0, 0, 0, 1],
    ])
    ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    vis.capture_screen_image(out_path, do_render=True)
    print(f"  → {os.path.basename(out_path)}")
    vis.destroy_window()


# ═══════════════════════════════════════════════════════════════════════════

print("=" * 55)
print("渲染 3D 结果 → PNG 截图  (v2 — 旋转网格法)")
print("=" * 55)

for sign in ["pos", "neg"]:
    render_mesh(os.path.join(DATA_DIR, f"final_mesh_{sign}.ply"), f"mesh_{sign}")

for sign in ["pos", "neg"]:
    render_cloud(os.path.join(DATA_DIR, f"final_cloud_{sign}.ply"), f"cloud_{sign}")

print(f"\n截图文件夹: {OUTPUT_DIR}")
print("完毕")
