"""
Final pipeline — pot + plant same rot_center transform per frame
"""
import numpy as np, cv2, open3d as o3d, os, json

DATA, OUT = r"E:\花瓶3d", os.path.join(r"E:\花瓶3d","output_v2")
os.makedirs(OUT,exist_ok=True)
FX,FY,CX,CY=365.456,365.456,254.878,205.395
ROT_CX,ROT_CZ,DEG,N=-0.0346,0.7861,10.0,36

def load_depth(i):
    r=np.fromfile(os.path.join(DATA,"depth",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_UNCHANGED)
def load_color_bgr(i):
    r=np.fromfile(os.path.join(DATA,"color",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_COLOR)
def rot_center(deg):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a)
    T1=np.eye(4);T1[0,3],T1[2,3]=-ROT_CX,-ROT_CZ
    R=np.eye(4);R[0,0],R[0,2],R[2,0],R[2,2]=c,s,-s,c
    T2=np.eye(4);T2[0,3],T2[2,3]=ROT_CX,ROT_CZ
    return T2@R@T1

mask_raw=np.fromfile(os.path.join(DATA,"output","mask_2d.png"),dtype=np.uint8)
mask_2d=cv2.imdecode(mask_raw,cv2.IMREAD_GRAYSCALE)
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
spatial_mask=cv2.dilate(mask_2d,k,iterations=1)

print("Per-frame extract + transform...")
all_pts,all_clr=[],[]
for i in range(N):
    depth=load_depth(i)
    df=cv2.bilateralFilter(depth.astype(np.float32),5,30,30)/1000.0
    bgr=load_color_bgr(i);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    vv,uu=np.mgrid[0:424,0:512];ok=(df>0.01)
    zv,uv,vv=df[ok],uu[ok],vv[ok]
    x=(uv-CX)*zv/FX;y=(vv-CY)*zv/FY;pts=np.stack([x,y,zv],axis=1)
    ur=np.clip((uv*1920//512).astype(int),0,1919)
    vr=np.clip((vv*1080//424).astype(int),0,1079)
    H=hsv[vr,ur];R=rgb[vr,ur]
    h,s,v=H[:,0].astype(float),H[:,1].astype(float),H[:,2].astype(float)
    r,g,b=R[:,0].astype(float),R[:,1].astype(float),R[:,2].astype(float)
    inm=spatial_mask[vr,ur]>0

    # Pot: 0.06<Y<0.20, not green
    is_green_simple=(h>30)&(h<90)&(s>25)
    is_pot=(inm & ~is_green_simple & (pts[:,1]>0.06)&(pts[:,1]<0.20)&(pts[:,2]>0.4)&(pts[:,2]<2.0))
    # Plant: Y<0.08, green
    is_plant=(inm & (h>=35)&(h<=85)&(s>30)&(v>30)&(g>r*0.95) & (pts[:,1]<0.08)&(pts[:,1]>-0.15)&(pts[:,2]>0.3)&(pts[:,2]<2.5))

    keep=is_pot|is_plant
    frame_pts=pts[keep];frame_clr=R[keep].astype(np.float64)/255.0

    # SAME transform for both
    T=rot_center(-i*DEG)
    hh=np.hstack([frame_pts,np.ones((len(frame_pts),1))])
    world=(T@hh.T).T[:,:3]
    all_pts.append(world);all_clr.append(frame_clr)
    if i%6==0:
        print(f"  Frame {i:03d}: pot={np.sum(is_pot)} plant={np.sum(is_plant)}")

# Merge + cleanup
print(f"\nMerge: {sum(len(p) for p in all_pts)} pts")
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(np.vstack(all_pts))
pcd.colors=o3d.utility.Vector3dVector(np.vstack(all_clr))
pcd=pcd.voxel_down_sample(0.003)
print(f"Voxel(3mm): {len(pcd.points)} pts")
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
print(f"SOR: {len(pcd.points)} pts")

# Save
o3d.io.write_point_cloud(os.path.join(OUT,"merged_cloud.ply"),pcd)
o3d.io.write_point_cloud(os.path.join(OUT,"merged_cloud.pcd"),pcd,write_ascii=True)
print(f"Saved: merged_cloud.ply ({len(pcd.points)} pts)")

poses={f"frame_{i:03d}":{"angle":round(i*DEG,1),"transform":rot_center(-i*DEG).tolist()} for i in range(N)}
with open(os.path.join(OUT,"pose.json"),"w") as f:json.dump(poses,f,indent=2)
with open(os.path.join(OUT,"report.json"),"w") as f:
    json.dump({"pipeline":"same_transform_per_frame","points":len(pcd.points)},f,indent=2)

# Viz
print("\nFINAL — close window.")
try:o3d.visualization.draw_geometries([pcd],window_name="Final",width=1280,height=720)
except Exception as e:print(f"Viz:{e}")
print("Done.")
