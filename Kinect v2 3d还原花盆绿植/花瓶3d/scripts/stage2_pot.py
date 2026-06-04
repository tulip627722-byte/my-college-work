"""
Stage 2: Pot — frame 24 ref, proper depth-to-color projection
Kinect V2 baseline ~5.2cm, color camera intrinsics for 1920x1080
"""
import numpy as np, cv2, open3d as o3d, os
from scipy.spatial import cKDTree as Tree

DATA,OUT=r"E:\花瓶3d",os.path.join(r"E:\花瓶3d","output_v2")
os.makedirs(OUT,exist_ok=True)

# Depth camera intrinsics (512x424)
FX_D,FY_D,CX_D,CY_D=365.456,365.456,254.878,205.395
# Color camera intrinsics (1920x1080) — Kinect V2 typical values
FX_C,FY_C,CX_C,CY_C=1060.0,1060.0,960.0,540.0
# Depth-to-color baseline (color cam ~5cm left of depth cam)
BASELINE=0.052  # meters

ROT_CX,ROT_CZ,DEG,N=-0.0346,0.7861,10.0,36
REF=24

def ld(i):
    r=np.fromfile(os.path.join(DATA,"depth",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_UNCHANGED)
def lc(i):
    r=np.fromfile(os.path.join(DATA,"color",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_COLOR)
def rc(d):
    a=np.radians(d);c,s=np.cos(a),np.sin(a)
    T1=np.eye(4);T1[0,3],T1[2,3]=-ROT_CX,-ROT_CZ
    R=np.eye(4);R[0,0],R[0,2],R[2,0],R[2,2]=c,s,-s,c
    T2=np.eye(4);T2[0,3],T2[2,3]=ROT_CX,ROT_CZ
    return T2@R@T1

mr=np.fromfile(os.path.join(DATA,"output","mask_2d.png"),dtype=np.uint8)
m2=cv2.imdecode(mr,cv2.IMREAD_GRAYSCALE)
sm=cv2.dilate(m2,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35)),iterations=1)

# ============================================================
# Pass 1: Geometry only, store camera-frame 3D position for color lookup
# ============================================================
print("Pass 1: Geometry assembly (storing cam-3D for color projection)")
all_geo=[]  # each: (world_pts, frame_id, cam_pts)
for i in range(N):
    df=cv2.bilateralFilter(ld(i).astype(np.float32),5,30,30)/1000.0
    bgr=lc(i);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    vv,uu=np.mgrid[0:424,0:512];ok=df>0.01
    zv,uv,vv=df[ok],uu[ok],vv[ok]
    x=(uv-CX_D)*zv/FX_D;y=(vv-CY_D)*zv/FY_D;pts=np.stack([x,y,zv],axis=1)
    ur=np.clip((uv*1920//512).astype(int),0,1919);vr=np.clip((vv*1080//424).astype(int),0,1079)
    H=hsv[vr,ur];h,s,v=H[:,0].astype(float),H[:,1].astype(float),H[:,2].astype(float)
    inm=sm[vr,ur]>0

    is_green=(h>30)&(h<90)&(s>25)
    is_pot=(inm & ~is_green & (pts[:,1]>0.08)&(pts[:,1]<0.20)&(pts[:,2]>0.4)&(pts[:,2]<2.0))

    cam_pts=pts[is_pot]  # 3D in depth camera coords
    T=rc((REF-i)*DEG)
    ph=np.hstack([cam_pts,np.ones((len(cam_pts),1))])
    world=(T@ph.T).T[:,:3]
    frame_id=np.full(len(world),i,dtype=np.int32)
    all_geo.append((world,frame_id,cam_pts))
    if i%6==0:print(f"  Frame {i:03d}: {len(world)} pot pts")

# Merge geometry
print("\nMerging geometry...")
all_world=np.vstack([g[0] for g in all_geo])
all_fid=np.concatenate([g[1] for g in all_geo])
all_cam=np.vstack([g[2] for g in all_geo])
print(f"  Raw: {len(all_world)} pts")

pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(all_world)
pcd=pcd.voxel_down_sample(0.002)
print(f"  Voxel(2mm): {len(pcd.points)} pts")
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
print(f"  SOR: {len(pcd.points)} pts")
geo_pts=np.asarray(pcd.points)
print(f"  Final: {len(geo_pts)} pts")

# ============================================================
# Pass 2: Color via proper depth-to-color 3D projection
# ============================================================
print("\nPass 2: Color via depth-to-color projection")
raw_tree=Tree(all_world)
colors=np.zeros((len(geo_pts),3),dtype=np.float64)
color_imgs=[lc(i) for i in range(N)]

def project_to_color(cam_xyz):
    """Project 3D point in depth camera coords to color image pixel"""
    Xd,Yd,Zd=cam_xyz[:,0],cam_xyz[:,1],cam_xyz[:,2]
    # Transform to color camera: color cam is BASELINE left of depth cam
    Xc=Xd+BASELINE;Yc=Yd;Zc=Zd
    uc=np.clip((FX_C*Xc/Zc+CX_C).astype(int),0,1919)
    vc=np.clip((FY_C*Yc/Zc+CY_C).astype(int),0,1079)
    return uc,vc

for gi in range(len(geo_pts)):
    pt=geo_pts[gi]
    dist,idx=raw_tree.query(pt,k=1)
    fid=all_fid[idx]
    cam=all_cam[idx].reshape(1,3)
    uc,vc=project_to_color(cam)
    bgr=color_imgs[fid][vc[0],uc[0]]
    colors[gi]=bgr[::-1].astype(np.float64)/255.0
    if gi%10000==0:print(f"  Colored {gi}/{len(geo_pts)}")

pcd.colors=o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(OUT,"pot_stage2.ply"),pcd)
print(f"\nSaved: pot_stage2.ply ({len(pcd.points)} pts)")

print("\nPOT — close window to confirm.")
try:o3d.visualization.draw_geometries([pcd],window_name="Pot ref=24 depth2color proj",width=1280,height=720)
except Exception as e:print(f"Viz:{e}")
print("Done.")
