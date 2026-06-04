"""View pot/plant separation"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Load the real separated point cloud (with actual colors preserved)
pcd_pot = o3d.io.read_point_cloud(os.path.join(OUT, "debug_frame000_pot.ply"))
pcd_plant = o3d.io.read_point_cloud(os.path.join(OUT, "debug_frame000_plant.ply"))

# Also load the labeled viz version
pcd_viz = o3d.io.read_point_cloud(os.path.join(OUT, "debug_frame000_separated.ply"))

print(f"Pot: {len(pcd_pot.points)} pts, Plant: {len(pcd_plant.points)} pts")

# Create wireframe cylinders to show rotation axis
rot_cx, rot_cz = -0.0346, 0.7969
axis_pts = []
for y in np.linspace(-0.1, 0.3, 20):
    axis_pts.append([rot_cx, y, rot_cz])
axis_pcd = o3d.geometry.PointCloud()
axis_pcd.points = o3d.utility.Vector3dVector(np.array(axis_pts))
axis_pcd.paint_uniform_color([1, 0, 0])  # Red axis line

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

print("Blue=pot (real color) | Green=plant (real color) | Red dots=rotation axis")
print("Left-drag=rotate | Scroll=zoom | R=reset | Close to continue")

try:
    o3d.visualization.draw_geometries(
        [pcd_pot, pcd_plant, axis_pcd, origin],
        window_name="Blue=Pot  Green=Plant  Red=Rotation Axis",
        width=1280, height=720,
    )
except Exception as e:
    print(f"Interactive failed: {e}")
    # Fallback: make 3-view matplotlib image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for label, pcd_obj, clr in [("Pot", pcd_pot, 'blue'), ("Plant", pcd_plant, 'green')]:
        pts = np.asarray(pcd_obj.points)
        if len(pts) > 3000:
            idx = np.random.choice(len(pts), 3000, replace=False)
            pts = pts[idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].scatter(pts[:,0], pts[:,1], c=clr, s=5, alpha=0.7)
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_title(f'{label} - Front')
        axes[0].axvline(rot_cx, color='red', ls='--', alpha=0.5)
        axes[0].set_aspect('equal')

        axes[1].scatter(pts[:,0], pts[:,2], c=clr, s=5, alpha=0.7)
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Z'); axes[1].set_title(f'{label} - Top')
        axes[1].scatter([rot_cx], [rot_cz], c='red', s=80, marker='x')
        axes[1].set_aspect('equal')

        axes[2].scatter(pts[:,2], pts[:,1], c=clr, s=5, alpha=0.7)
        axes[2].set_xlabel('Z'); axes[2].set_ylabel('Y'); axes[2].set_title(f'{label} - Side')
        axes[2].set_aspect('equal')

        plt.suptitle(f'{label} points', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"debug_frame000_{label.lower()}_views.png"), dpi=150)
        print(f"Saved: debug_frame000_{label.lower()}_views.png")
    plt.close('all')
