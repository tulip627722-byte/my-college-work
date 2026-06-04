"""Show cropped frame 0 — try interactive, fallback to images"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

pcd = o3d.io.read_point_cloud(os.path.join(OUT, "debug_frame000_manual.ply"))
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"Points: {len(pts)}")
print(f"X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
print(f"Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
print(f"Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")

# Try interactive
try:
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    print("Launching interactive viewer...")
    print("  Left-drag=rotate  Scroll=zoom  Right-drag=pan")
    o3d.visualization.draw_geometries(
        [pcd, origin],
        window_name="Cropped Frame 0 - Verify selection",
        width=1280, height=720,
    )
except Exception as e:
    print(f"Interactive failed: {e}")
    print("Falling back to image rendering...")

    # Matplotlib fallback
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Random sample for speed
    idx = np.random.choice(len(pts), min(len(pts), 5000), replace=False)
    xs, ys, zs = pts[idx,0], pts[idx,1], pts[idx,2]
    cs = colors[idx] if pcd.has_colors() else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].scatter(xs, ys, c=cs if cs is not None else zs, s=5, alpha=0.9)
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Front'); axes[0].set_aspect('equal')

    axes[1].scatter(xs, zs, c=cs if cs is not None else ys, s=5, alpha=0.9)
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Z (m)')
    axes[1].set_title('Top'); axes[1].set_aspect('equal')

    axes[2].scatter(zs, ys, c=cs if cs is not None else xs, s=5, alpha=0.9)
    axes[2].set_xlabel('Z (m)'); axes[2].set_ylabel('Y (m)')
    axes[2].set_title('Side'); axes[2].set_aspect('equal')

    plt.suptitle('Cropped Frame 0', fontsize=14)
    plt.tight_layout()
    img_path = os.path.join(OUT, "debug_frame000_manual_views.png")
    plt.savefig(img_path, dpi=150)
    print(f"Saved: {img_path}")
    print("Open this PNG to verify the cropped point cloud.")
