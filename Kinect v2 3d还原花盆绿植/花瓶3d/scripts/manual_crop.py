"""手动框选花盆区域 - Open3D VisualizerWithEditing"""
import open3d as o3d
import numpy as np
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# 读带颜色的帧0（隔点采样后的版本，交互更流畅）
ply_path = os.path.join(OUT, "all_frames", "frame_000.ply")
if not os.path.exists(ply_path):
    # fallback
    ply_path = os.path.join(OUT, "debug_frame000_raw.ply")

pcd = o3d.io.read_point_cloud(ply_path)
print(f"加载: {ply_path}")
print(f"点数: {len(pcd.points)}")

# 如果原来有颜色就保留，否则涂绿
if not pcd.has_colors():
    pcd.paint_uniform_color([0.3, 0.8, 0.3])

print()
print("=" * 50)
print("操作说明：")
print("  Shift + 鼠标左键拖拽  =  框选要保留的点")
print("  Shift + 鼠标左键点击  =  单选点")
print("  Ctrl + Z              =  撤销上次选择")
print("  关闭窗口后自动保存")
print("=" * 50)
print()
print("请框选【花盆 + 绿植】区域（不要桌面/背景/底座）")
print()

try:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Manual Crop - Select vase+plant region", width=1280, height=900)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    print(f"\n选中点数: {len(picked)}")

    if len(picked) > 0:
        cropped = pcd.select_by_index(picked)
        out_path = os.path.join(OUT, "debug_frame000_manual.ply")
        o3d.io.write_point_cloud(out_path, cropped)
        print(f"已保存: {out_path}")

        # Also print the bounding box for batch processing
        pts = np.asarray(cropped.points)
        print(f"\n裁剪后点云范围 (可用于批量处理):")
        print(f"  X: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
        print(f"  Y: [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")
        print(f"  Z: [{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")
        print(f"\n✅ 告诉我继续，我用这个范围自动裁剪全部36帧")
    else:
        print("⚠️ 未选中任何点！请重新运行并框选。")

except Exception as e:
    print(f"\n❌ 窗口启动失败: {e}")
    print("\n改用方案B：你直接告诉我大概的XYZ范围，我帮你切。")
    print("参考：花盆大概在 X:-0.5~0.5, Y:-0.5~1.0, Z:0.5~2.0 左右")
