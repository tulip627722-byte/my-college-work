import open3d as o3d
import numpy as np

def get_rotation_center_manually(pcd_path):
    print(f"正在加载点云: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print("点云为空，请检查路径！")
        return

    print("\n" + "="*50)
    print("【手动标定操作指南】")
    print("1. 在弹出的窗口中，旋转/平移视角，找到花盆的圆形边缘（建议选盆口沿）。")
    print("2. 按住 [Shift] + 鼠标左键，在圆周上尽量等距地选择 3 个点。")
    print("3. 选完 3 个点后，按 [Q] 键退出窗口，系统将自动计算圆心。")
    print("="*50 + "\n")

    # 激活可视化编辑窗口
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="手动标定旋转中心 (按Shift+左键选点, 按Q退出)", width=1024, height=768)
    vis.add_geometry(pcd)
    vis.run()  # 阻塞程序，等待用户操作
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    if len(picked_indices) != 3:
        print(f"⚠️ 标定失败：您选择了 {len(picked_indices)} 个点。必须精确选择 3 个点才能确定一个圆心！")
        return

    # 提取用户选择的三个点的物理坐标
    points = np.asarray(pcd.points)
    p1, p2, p3 = points[picked_indices[0]], points[picked_indices[1]], points[picked_indices[2]]

    # 提取 X 和 Z 坐标 (我们在 XZ 平面上进行圆拟合)
    x1, z1 = p1[0], p1[2]
    x2, z2 = p2[0], p2[2]
    x3, z3 = p3[0], p3[2]

    # 解线性方程组计算三角形外心（即外接圆圆心）
    A = np.array([
        [2 * (x2 - x1), 2 * (z2 - z1)],
        [2 * (x3 - x2), 2 * (z3 - z2)]
    ])
    B = np.array([
        x2**2 + z2**2 - x1**2 - z1**2,
        x3**2 + z3**2 - x2**2 - z2**2
    ])

    try:
        center_xz = np.linalg.solve(A, B)
        avg_y = (p1[1] + p2[1] + p3[1]) / 3.0  # Y轴取三个点的平均高度
        center = np.array([center_xz[0], avg_y, center_xz[1]])
        
        print("\n✅ 旋转圆心计算成功！")
        print("\n>>> 请将你拼接代码中的 CENTER 参数替换为以下内容 <<<")
        print(f"CENTER = np.array([{center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f}])\n")
        
        return center
    except np.linalg.LinAlgError:
        print("⚠️ 计算失败：这三个点可能在一条直线上，无法构成圆，请重新运行脚本选择。")
        return None

if __name__ == "__main__":
    # 请确保路径指向你的第一帧点云数据
    get_rotation_center_manually("plant_pure_0001.ply")