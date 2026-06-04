"""
Stage 3: Plant — frame 2 seed, green-ratio filter, dedup
Frame 2 = identity (camera-2 coords), other frames register via nominal rotation
Green ratio: greenness = saturation * (green/total_rgb), filter weak green
"""
import numpy as np, cv2, open3d as o3d, os
from scipy.spatial import cKDTree

DATA,OUT=r"E:\花瓶3d",os.path.join(r"E:\花瓶3d","output_v2")
os.makedirs(OUT,exist_ok=True)
FX,FY,CX,CY=365.456,365.456,254.878,205.395
ROT_CX,ROT_CZ,DEG,N=-0.0346,0.7861,10.0,36
REF=2  # frame 2 = seed

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
# Extract plant points per frame with greenness score
# ============================================================
print("Extracting plant points with greenness filter...")
plant_frames=[]  # list of (pts_cam, clr, greenness) per frame in camera coords

for i in range(N):
    df=cv2.bilateralFilter(ld(i).astype(np.float32),5,30,30)/1000.0
    bgr=lc(i);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    vv,uu=np.mgrid[0:424,0:512];ok=df>0.01
    zv,uv,vv=df[ok],uu[ok],vv[ok]
    x=(uv-CX)*zv/FX;y=(vv-CY)*zv/FY;pts=np.stack([x,y,zv],axis=1)
    ur=np.clip((uv*1920//512).astype(int),0,1919);vr=np.clip((vv*1080//424).astype(int),0,1079)
    H=hsv[vr,ur];R=rgb[vr,ur]
    h,s,v=H[:,0].astype(float),H[:,1].astype(float),H[:,2].astype(float)
    r,g,b=R[:,0].astype(float),R[:,1].astype(float),R[:,2].astype(float)
    inm=sm[vr,ur]>0

    # Plant: Y<0.08, green detection
    is_plant=(inm & (h>=35)&(h<=85)&(s>30)&(v>30)&(g>r*0.95) &
              (pts[:,1]<0.08)&(pts[:,1]>-0.15)&(pts[:,2]>0.3)&(pts[:,2]<2.5))

    pt=pts[is_plant];cl=R[is_plant].astype(np.float64)/255.0
    # Greenness score: higher = more confidently green
    gs=s[is_plant]/255.0 * (g[is_plant]/(r[is_plant]+g[is_plant]+b[is_plant]+0.001))

    plant_frames.append((pt,cl,gs))
    if i%6==0:print(f"  Frame {i:03d}: {len(pt)} plant pts (greenness {gs.mean():.3f})")

# ============================================================
# Frame REF = seed, then add from other frames with green-ratio dedup
# ============================================================
print(f"\nFrame {REF} = seed. Adding frames with dedup + green check...")

# Frame order: start from REF, then spiral out
order=list(range(REF,N))+list(range(0,REF))
print(f"  Order: {order}")

model_pts=None;model_clr=None;model_gs=None

for idx,j in enumerate(order):
    pt_j,cl_j,gs_j=plant_frames[j]

    # Transform to camera-REF coords
    T=rc((REF-j)*DEG)
    ph=np.hstack([pt_j,np.ones((len(pt_j),1))])
    world=(T@ph.T).T[:,:3]

    if idx==0:
        # Seed
        model_pts=world;model_clr=cl_j;model_gs=gs_j
        print(f"  Seed frame {j}: {len(model_pts)} pts")
        continue

    # Dedup against model
    tree=cKDTree(model_pts)
    dist,nn_idx=tree.query(world,k=1,distance_upper_bound=0.003)

    # New points (no neighbor within 3mm)
    is_new=~np.isfinite(dist)|(dist>=0.003)
    # Overlapping points (within 3mm) — check greenness
    is_overlap=np.isfinite(dist)&(dist<0.003)

    n_new=np.sum(is_new)
    n_overlap=np.sum(is_overlap)

    if n_new>0:
        model_pts=np.vstack([model_pts,world[is_new]])
        model_clr=np.vstack([model_clr,cl_j[is_new]])
        model_gs=np.concatenate([model_gs,gs_j[is_new]])

    # For overlaps: if new frame sees it greener, replace color
    n_upgraded=0
    if n_overlap>0:
        ol_idx=nn_idx[is_overlap]
        ol_gs_new=gs_j[is_overlap]
        ol_gs_old=model_gs[ol_idx]
        upgrade=ol_gs_new>ol_gs_old
        if np.sum(upgrade)>0:
            up_idx=ol_idx[upgrade]
            model_clr[up_idx]=cl_j[is_overlap][upgrade]
            model_gs[up_idx]=ol_gs_new[upgrade]
            n_upgraded=np.sum(upgrade)

    print(f"  Frame {j:03d}: +{n_new} new, {n_overlap} overlap ({n_upgraded} upgraded) | total={len(model_pts)}")

# ============================================================
# Final greenness filter: remove points with low greenness
# ============================================================
GREEN_THRESH=0.15  # points with greenness below this are likely shadows
keep=model_gs>=GREEN_THRESH
print(f"\nGreenness filter (thresh={GREEN_THRESH}): {np.sum(keep)}/{len(model_pts)} pts kept ({np.sum(~keep)} removed)")

model_pts=model_pts[keep];model_clr=model_clr[keep]

# Voxel cleanup
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(model_pts);pcd.colors=o3d.utility.Vector3dVector(model_clr)
pcd=pcd.voxel_down_sample(0.002)
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=3.0)
print(f"After voxel+SOR: {len(pcd.points)} pts")

o3d.io.write_point_cloud(os.path.join(OUT,"plant_stage3.ply"),pcd)
print(f"Saved: plant_stage3.ply ({len(pcd.points)} pts)")

# Viz
print("\nPLANT — close window to confirm.")
try:
    o3d.visualization.draw_geometries([pcd],window_name=f"Plant Stage3 (ref=frame{REF})",width=1280,height=720)
except Exception as e:print(f"Viz:{e}")
print("Plant done.")
