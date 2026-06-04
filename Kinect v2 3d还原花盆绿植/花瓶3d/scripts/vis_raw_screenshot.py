"""Render frame 0 raw point cloud to PNG screenshots (offscreen)"""
import numpy as np
import cv2
import open3d as o3d
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Read depth
depth_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(depth_raw, cv2.IMREAD_UNCHANGED)

h, w = depth.shape
fx = fy = 365.456
cx, cy = 254.878, 205.395

v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy
pts = np.stack([x, y, zv], axis=1)

# Read color & map
color_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_bgr = cv2.imdecode(color_raw, cv2.IMREAD_COLOR)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
colors = color_rgb[vv, uv] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Offscreen renderer
render = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
render.scene.set_background([0.1, 0.1, 0.1, 1.0])

mtl = o3d.visualization.rendering.MaterialRecord()
mtl.shader = "defaultUnlit"
mtl.point_size = 3.0
render.scene.add_geometry("pcd", pcd, mtl)

# View 1: front
render.setup_camera(60, pcd.get_center(), pcd.get_center() + [0, 0, 3], [0, -1, 0])
img = render.render_to_image()
o3d.io.write_image(os.path.join(OUT, "debug_frame000_raw_front.png"), img)
print("Saved: debug_frame000_raw_front.png")

# View 2: top
render.setup_camera(60, pcd.get_center(), pcd.get_center() + [0, -5, 0.1], [0, 0, 1])
img2 = render.render_to_image()
o3d.io.write_image(os.path.join(OUT, "debug_frame000_raw_top.png"), img2)
print("Saved: debug_frame000_raw_top.png")

# View 3: side
render.setup_camera(60, pcd.get_center(), pcd.get_center() + [5, 0, 0.5], [0, -1, 0])
img3 = render.render_to_image()
o3d.io.write_image(os.path.join(OUT, "debug_frame000_raw_side.png"), img3)
print("Saved: debug_frame000_raw_side.png")

print("Done. Check output/ folder for screenshots.")
