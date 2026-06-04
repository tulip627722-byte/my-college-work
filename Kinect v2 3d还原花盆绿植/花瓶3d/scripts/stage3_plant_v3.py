"""
Stage 3 v3: Plant — dual seed (frame 0 + frame 35), incremental ICP
"""
import numpy as np
import cv2
import open3d as o3d
import os

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
    a=np.radians(deg); c,s=np.cos(a),np.sin(a)
    T1=np.eye(4); T1[0,3]=-ROT_CX; T1[2,3]=-ROT_CZ
    R=np.eye(4); R[0,0]=c; R[0,2]=s; R[2,0]=-s; R[2,2]=c
    T2=np.eye(4); T2[0,3]=ROT_CX; T2[2,3]=ROT_CZ
    return T2@R@T1

mask_raw=np.fromfile(os.path.join(DATA,"output","mask_2d.png"),dtype=np.uint8)
mask_2d=cv2.imdecode(mask_raw,cv2.IMREAD_GRAYSCALE)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
spatial_mask=cv2.dilate(mask_2d,kernel,iterations=1)

# ============================================================
# Extract plant clouds per frame
# ============================================================
print("Extracting plant clouds per frame...")
plant_pcds = []
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
    r,g,b=rgb[:,0].astype(float),rgb[:,1].astype(float),rgb[:,2].astype(float)
    in_m=spatial_mask[vr,ur]>0

    is_plant_hsv = (h >= 35) & (h <= 85) & (s > 30) & (v > 30)
    is_plant_rgb = g > r * 0.95
    is_plant = (in_m & is_plant_hsv & is_plant_rgb &
                (pts[:,1] < 0.10) & (pts[:,1] > -0.15) &
                (pts[:,2] > 0.3) & (pts[:,2] < 2.5))

    plant_pts = pts[is_plant]
    plant_rgb = rgb[is_plant]

    if len(plant_pts) < 30:
        plant_pcds.append(None)
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plant_pts)
    pcd.colors = o3d.utility.Vector3dVector(plant_rgb.astype(np.float64)/255.0)
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    plant_pcds.append(pcd)
    if i%6==0: print(f"  Frame {i:03d}: {len(pcd.points)} plant pts")

# ============================================================
# Dual seed: frame 0 (identity) + frame 35 (rotated to align with 0)
# ============================================================
print("\nBuilding dual-seed model (frame 0 + frame 35)...")

# Frame 0 at identity
seed0 = plant_pcds[0]
T35_to_world = rot_center(-35 * DEG)  # undo 350deg turntable rotation
seed35 = o3d.geometry.PointCloud(plant_pcds[35])
seed35.transform(T35_to_world)

world_cloud = seed0 + seed35
world_cloud = world_cloud.voxel_down_sample(0.002)
print(f"  Dual seed: {len(world_cloud.points)} pts")

# ============================================================
# Forward pass: frames 1->34, ICP to accumulated model
# ============================================================
print("\nForward ICP (frames 1-34) -> accumulated model...")
for i in range(1, 35):
    src = plant_pcds[i]
    if src is None:
        continue

    init = rot_center(-i * DEG)
    try:
        reg = o3d.pipelines.registration.registration_icp(
            src, world_cloud, 0.01, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        T = reg.transformation
        fit, rmse = reg.fitness, reg.inlier_rmse
    except:
        T = init; fit, rmse = 0, 0

    src_w = o3d.geometry.PointCloud(src)
    src_w.transform(T)
    world_cloud += src_w
    if i % 3 == 0:
        world_cloud = world_cloud.voxel_down_sample(0.002)

    if i%6==0 or i==34:
        print(f"  Frame {i:03d}: fit={fit:.3f} rmse={rmse:.4f} | model={len(world_cloud.points)} pts")

# Final cleanup
world_cloud = world_cloud.voxel_down_sample(0.002)
world_cloud, _ = world_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
print(f"\nFinal plant cloud: {len(world_cloud.points)} pts")
o3d.io.write_point_cloud(os.path.join(OUT,"plant_merged.ply"), world_cloud)

# ============================================================
# Poisson mesh
# ============================================================
print("\nPoisson reconstruction...")
world_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=40))
world_cloud.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    world_cloud, depth=8, width=0, scale=1.1, linear_fit=False)

densities = np.asarray(densities)
d_min, d_max = densities.min(), densities.max()
mesh.remove_vertices_by_mask(densities < d_min + 0.05*(d_max-d_min))
mesh = mesh.simplify_vertex_clustering(0.002, contraction=o3d.geometry.SimplificationContraction.Average)
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
verts = np.asarray(mesh.vertices)
fin = np.all(np.isfinite(verts), axis=1)
if not fin.all(): mesh.remove_vertices_by_mask(~fin)
mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
mesh.compute_vertex_normals()

# Color
from scipy.spatial import cKDTree
pcd_pts = np.asarray(world_cloud.points)
pcd_clr = np.asarray(world_cloud.colors)
tree = cKDTree(pcd_pts)
_, nn_idx = tree.query(np.asarray(mesh.vertices), k=1)
mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_clr[nn_idx])

print(f"Plant mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
o3d.io.write_triangle_mesh(os.path.join(OUT,"plant_mesh.ply"), mesh)

# ============================================================
# Visualize
# ============================================================
print("\nPLANT CLOUD (dual seed) - close to continue.")
o3d.visualization.draw_geometries(
    [world_cloud], window_name="Stage 3 - Plant Dual-Seed Cloud",
    width=1280, height=720)

print("PLANT MESH - close to continue.")
o3d.visualization.draw_geometries(
    [mesh], window_name="Stage 3 - Plant Mesh (dual seed)",
    width=1280, height=720, mesh_show_back_face=True)

# Combined: write file first, then try to show
pot_mesh = o3d.io.read_triangle_mesh(os.path.join(OUT,"pot_mesh_color.ply"))
combined = pot_mesh + mesh
combined.remove_duplicated_vertices()
combined.remove_degenerate_triangles()
o3d.io.write_triangle_mesh(os.path.join(OUT,"combined_preview.ply"), combined)
print(f"Combined: {len(combined.vertices)} verts, {len(combined.triangles)} tris")
print("Saved: combined_preview.ply")

print("\nPOT + PLANT COMBINED - close to finish.")
try:
    o3d.visualization.draw_geometries(
        [combined], window_name="Stage 3 - Pot + Plant (dual seed)",
        width=1280, height=720, mesh_show_back_face=True)
except:
    print("GLFW window failed (known issue), but file saved.")

print("Stage 3 done.")
