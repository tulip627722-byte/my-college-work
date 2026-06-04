"""
Stage 2b: Add color to pot mesh + show colored merged point cloud
"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")

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

print("Extracting pot clouds WITH COLOR from all frames...")
all_pts, all_clr = [], []
for i in range(N):
    depth=load_depth(i)
    depth_bf=cv2.bilateralFilter(depth.astype(np.float32),5,30,30)/1000.0
    color_bgr=load_color_bgr(i)
    color_hsv=cv2.cvtColor(color_bgr,cv2.COLOR_BGR2HSV)
    color_rgb=cv2.cvtColor(color_bgr,cv2.COLOR_BGR2RGB)

    vv,uu=np.mgrid[0:424,0:512]; zz=depth_bf; ok=zz>0.01
    zv,uv,vv=zz[ok],uu[ok],vv[ok]
    x=(uv-CX)*zv/FX; y=(vv-CY)*zv/FY; pts=np.stack([x,y,zv],axis=1)

    ur=np.clip((uv*1920//512).astype(int),0,1919)
    vr=np.clip((vv*1080//424).astype(int),0,1079)
    hsv=color_hsv[vr,ur]; rgb=color_rgb[vr,ur]
    h,s,v=hsv[:,0].astype(float),hsv[:,1].astype(float),hsv[:,2].astype(float)
    in_m=spatial_mask[vr,ur]>0

    is_green = (h>30)&(h<90)&(s>25)
    is_pot = (in_m & (~is_green) &
              (pts[:,1]>0.06) & (pts[:,1]<0.20) &
              (pts[:,2]>0.4) & (pts[:,2]<2.0))

    pot_pts = pts[is_pot]
    pot_rgb = rgb[is_pot].astype(np.float64) / 255.0

    # Transform to world frame
    T = rot_center(-i*DEG)
    pot_h = np.hstack([pot_pts, np.ones((len(pot_pts),1))])
    pot_world = (T @ pot_h.T).T[:, :3]

    all_pts.append(pot_world)
    all_clr.append(pot_rgb)
    if i%6==0: print(f"  Frame {i:03d}: {len(pot_world)} pts")

# Merge
merged_pts = np.vstack(all_pts)
merged_clr = np.vstack(all_clr)
print(f"Merged: {len(merged_pts)} pts")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_pts)
pcd.colors = o3d.utility.Vector3dVector(merged_clr)
pcd = pcd.voxel_down_sample(0.002)
print(f"Voxel(2mm): {len(pcd.points)} pts")

pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print(f"SOR: {len(pcd.points)} pts")

pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=40))
pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

# Save colored merged cloud
o3d.io.write_point_cloud(os.path.join(OUT,"pot_merged_color.ply"), pcd)
print("Saved: pot_merged_color.ply")

# ============================================================
# Load existing mesh, transfer color from colored point cloud
# ============================================================
print("\nLoading pot mesh and transferring colors...")
mesh = o3d.io.read_triangle_mesh(os.path.join(OUT,"pot_mesh.ply"))
# Remove non-finite vertices
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()
mesh_verts = np.asarray(mesh.vertices)
# Filter out NaN/inf vertices
finite_mask = np.all(np.isfinite(mesh_verts), axis=1)
if not finite_mask.all():
    print(f"  Removing {np.sum(~finite_mask)} non-finite vertices")
    mesh.remove_vertices_by_mask(~finite_mask)
    mesh_verts = np.asarray(mesh.vertices)
print(f"Mesh: {len(mesh_verts)} verts, {len(mesh.triangles)} tris")

# Build KDTree on colored point cloud
pcd_pts = np.asarray(pcd.points)
pcd_clr = np.asarray(pcd.colors)

# For each mesh vertex, find nearest colored point using scipy KDTree
print("  Nearest-neighbor color transfer...")
from scipy.spatial import cKDTree
tree = cKDTree(pcd_pts)
_, nn_idx = tree.query(mesh_verts, k=1)
mesh_colors = pcd_clr[nn_idx]

mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(OUT,"pot_mesh_color.ply"), mesh)
print("Saved: pot_mesh_color.ply")

# ============================================================
# Visualize
# ============================================================
print("\nCOLORED MERGED CLOUD — close to see mesh.")
o3d.visualization.draw_geometries(
    [pcd], window_name="Stage 2 — Pot Merged Cloud (COLORED)",
    width=1280, height=720)

print("COLORED POT MESH — close window to finish.")
o3d.visualization.draw_geometries(
    [mesh], window_name="Stage 2 — Pot Mesh (COLORED)",
    width=1280, height=720, mesh_show_back_face=True)

print("Done.")
