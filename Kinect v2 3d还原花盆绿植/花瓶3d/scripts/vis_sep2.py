"""Pot/plant viz — blue pot, green plant, projections"""
import numpy as np
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Use labeled version: blue=pot, green=plant
pcd = o3d.io.read_point_cloud(os.path.join(OUT, "debug_frame000_separated.ply"))
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Separate by color label for stats
blue_mask = colors[:, 2] > 0.6  # B channel > 0.6 => pot
green_mask = colors[:, 1] > 0.6  # G channel > 0.6 => plant

print(f"Pot (blue):   {np.sum(blue_mask)} pts, Y=[{pts[blue_mask,1].min():.3f}, {pts[blue_mask,1].max():.3f}]")
print(f"Plant (green): {np.sum(green_mask)} pts, Y=[{pts[green_mask,1].min():.3f}, {pts[green_mask,1].max():.3f}]")

# Rotation axis
rot_cx, rot_cz = -0.0346, 0.7969
axis_line = np.array([[rot_cx, y, rot_cz] for y in np.linspace(-0.1, 0.3, 30)])
pcd_axis = o3d.geometry.PointCloud()
pcd_axis.points = o3d.utility.Vector3dVector(axis_line)
pcd_axis.paint_uniform_color([1, 0, 0])
pcd_axis = pcd_axis.voxel_down_sample(0.01)  # keep as-is since it's a line

# Merge all
all_pts = np.vstack([pts, axis_line])
all_colors = np.vstack([colors, np.tile([1,0,0], (len(axis_line), 1))])

pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(all_pts)
pcd_all.colors = o3d.utility.Vector3dVector(all_colors)

# Try interactive
print("Launching viewer... Blue=pot, Green=plant, Red=rotation axis")
try:
    o3d.visualization.draw_geometries(
        [pcd_all],
        window_name="BLUE=Pot  GREEN=Plant  RED=Axis",
        width=1280, height=720,
    )
except Exception as e:
    print(f"Window failed: {e}")

# Always save projections as backup
print("Saving projection images...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Subsample for speed
N = min(5000, len(pts))
idx = np.random.choice(len(pts), N, replace=False)
xs, ys, zs = pts[idx,0], pts[idx,1], pts[idx,2]
cs = colors[idx]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, (h_axis, v_axis, h_label, v_label, title) in zip(axes, [
    (0, 1, 'X (m)', 'Y (m)', 'FRONT (X vs Y)'),
    (0, 2, 'X (m)', 'Z (m)', 'TOP (X vs Z)'),
    (2, 1, 'Z (m)', 'Y (m)', 'SIDE (Z vs Y)'),
]):
    ax.scatter(pts[:, h_axis], pts[:, v_axis], c=colors, s=1, alpha=0.8)
    ax.set_xlabel(h_label); ax.set_ylabel(v_label)
    ax.set_title(title)
    # Rotation axis
    if h_axis == 0 and v_axis == 2:  # Top view: mark center
        ax.scatter([rot_cx], [rot_cz], c='red', s=100, marker='+', linewidths=2)
    elif h_axis == 2 and v_axis == 1:  # Side: mark Z
        ax.axvline(rot_cz, color='red', ls='--', alpha=0.5, linewidth=1)
    elif h_axis == 0 and v_axis == 1:  # Front
        ax.axvline(rot_cx, color='red', ls='--', alpha=0.5, linewidth=1)
    ax.set_aspect('equal')

plt.suptitle('BLUE = Pot  |  GREEN = Plant  |  Red cross/line = Rotation axis', fontsize=14)
plt.tight_layout()
img_path = os.path.join(OUT, "debug_frame000_separated_views.png")
plt.savefig(img_path, dpi=200)
print(f"Saved: {img_path}")
print("Open debug_frame000_separated_views.png to verify pot/plant separation.")
