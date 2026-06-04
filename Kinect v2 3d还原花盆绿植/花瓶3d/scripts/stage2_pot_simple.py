"""
Stage 2: Pot assembly — direct 10deg/frame rotation, no ICP, no TSDF
"""
import numpy as np
import cv2
import open3d as o3d
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output_v2")
os.makedirs(OUT, exist_ok=True)

FX,FY=365.456,365.456; CX,CY=254.878,205.395
ROT_CX,ROT_CZ=-0.0346,0.7861; DEG=360.0/36; N=36

def load_depth(i):
    raw=np.fromfile(os.path.join(DATA,"depth",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(raw,cv2.IMREAD_UNCHANGED)
def load_color_bgr(i):
    raw=np.fromfile(os.path.join(DATA,"color",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(raw,cv2.IMREAD_COLOR)

def rot_center(deg):
    """Rotation around Y axis at (ROT_CX, 0, ROT_CZ)"""
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

print("Extracting pot clouds from all 36 frames...")
all_pot = []
for i in range(N):
    depth=load_depth(i)
    depth_bf=cv2.bilateralFilter(depth.astype(np.float32),5,30,30)/1000.0
    color_bgr=load_color_bgr(i)
    color_hsv=cv2.cvtColor(color_bgr,cv2.COLOR_BGR2HSV)

    vv,uu=np.mgrid[0:424,0:512]; zz=depth_bf; ok=zz>0.01
    zv,uv,vv=zz[ok],uu[ok],vv[ok]
    x=(uv-CX)*zv/FX; y=(vv-CY)*zv/FY; pts=np.stack([x,y,zv],axis=1)

    ur=np.clip((uv*1920//512).astype(int),0,1919)
    vr=np.clip((vv*1080//424).astype(int),0,1079)
    hsv=color_hsv[vr,ur]; h,s,v=hsv[:,0].astype(float),hsv[:,1].astype(float),hsv[:,2].astype(float)
    in_m=spatial_mask[vr,ur]>0

    is_green = (h>30)&(h<90)&(s>25)
    is_pot = (in_m & (~is_green) &
              (pts[:,1]>0.06) & (pts[:,1]<0.20) &
              (pts[:,2]>0.4) & (pts[:,2]<2.0))

    pot_pts = pts[is_pot]

    # Transform to world frame: rotate by -i*DEG to align with frame 0
    T = rot_center(-i*DEG)
    pot_h = np.hstack([pot_pts, np.ones((len(pot_pts),1))])
    pot_world = (T @ pot_h.T).T[:, :3]

    all_pot.append(pot_world)
    if i%6==0:
        print(f"  Frame {i:03d}: {len(pot_world)} pot pts")

# Merge
print("\nMerging all pot point clouds...")
merged_pts = np.vstack(all_pot)
print(f"  Before downsample: {len(merged_pts)} pts")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_pts)
pcd = pcd.voxel_down_sample(0.002)
print(f"  After voxel(2mm): {len(pcd.points)} pts")

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print(f"  After SOR: {len(pcd.points)} pts")

# Estimate normals for mesh reconstruction
print("\nEstimating normals...")
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=40))
pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

# Save merged cloud
o3d.io.write_point_cloud(os.path.join(OUT,"pot_merged.ply"), pcd)
print(f"Saved: pot_merged.ply ({len(pcd.points)} pts)")

# Poisson mesh
print("\nPoisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, width=0, scale=1.1, linear_fit=False)

# Remove low-density faces
densities = np.asarray(densities)
d_min, d_max = densities.min(), densities.max()
threshold = d_min + 0.1 * (d_max - d_min)
verts_to_remove = densities < threshold
mesh.remove_vertices_by_mask(verts_to_remove)

mesh = mesh.simplify_vertex_clustering(0.002, contraction=o3d.geometry.SimplificationContraction.Average)
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh = mesh.filter_smooth_taubin(number_of_iterations=3)
mesh.compute_vertex_normals()

print(f"Pot mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT,"pot_mesh.ply"), mesh)

# Visualize
print("\nPOT MESH — check then close window to continue.")
o3d.visualization.draw_geometries(
    [mesh], window_name="Stage 2 — Pot Mesh (direct 10deg assembly)",
    width=1280, height=720, mesh_show_back_face=True)

# Also show merged point cloud for comparison
pcd_viz = pcd.voxel_down_sample(0.004)
print("MERGED CLOUD — close to finish.")
o3d.visualization.draw_geometries(
    [pcd_viz], window_name="Stage 2 — Pot Merged Cloud",
    width=1280, height=720)

print("Stage 2 done.")
