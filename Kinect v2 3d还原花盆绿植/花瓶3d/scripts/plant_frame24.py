"""
Frame 24 plant from manual mask -> 3D point cloud
"""
import numpy as np, cv2, open3d as o3d, os

DATA,OUT=r"E:\花瓶3d",os.path.join(r"E:\花瓶3d","output_v2")
os.makedirs(OUT,exist_ok=True)
FX,FY,CX,CY=365.456,365.456,254.878,205.395
BASELINE=0.052;FX_C,FY_C,CX_C,CY_C=1060.0,1060.0,960.0,540.0

def ld(i):
    r=np.fromfile(os.path.join(DATA,"depth",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_UNCHANGED)
def lc(i):
    r=np.fromfile(os.path.join(DATA,"color",f"{i:03d}.png"),dtype=np.uint8)
    return cv2.imdecode(r,cv2.IMREAD_COLOR)

# Load manual mask
mr=np.fromfile(os.path.join(DATA,"output","mask_plant_frame24.png"),dtype=np.uint8)
mask_plant=cv2.imdecode(mr,cv2.IMREAD_GRAYSCALE)

FRAME=24
print(f"Frame {FRAME} plant from manual mask")

df=cv2.bilateralFilter(ld(FRAME).astype(np.float32),5,30,30)/1000.0
bgr=lc(FRAME);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
vv,uu=np.mgrid[0:424,0:512];ok=df>0.01
zv,uv,vv=df[ok],uu[ok],vv[ok]
x=(uv-CX)*zv/FX;y=(vv-CY)*zv/FY;pts=np.stack([x,y,zv],axis=1)
ur=np.clip((uv*1920//512).astype(int),0,1919);vr=np.clip((vv*1080//424).astype(int),0,1079)
H=hsv[vr,ur];R=rgb[vr,ur]
h,s,v=H[:,0].astype(float),H[:,1].astype(float),H[:,2].astype(float)
r,g,b=R[:,0].astype(float),R[:,1].astype(float),R[:,2].astype(float)

# Plant: manual mask only + Y separation + valid depth
in_mask=mask_plant[vr,ur]>0
is_plant=(in_mask &
          (pts[:,1]<0.08)&(pts[:,1]>-0.15)&  # Y separation from pot
          (pts[:,2]>0.3)&(pts[:,2]<3.0))     # depth range

plant_pts=pts[is_plant]
print(f"Manual mask plant: {len(plant_pts)} pts (mask only, no green filter)")

# Color via depth-to-color projection
Xd,Yd,Zd=plant_pts[:,0],plant_pts[:,1],plant_pts[:,2]
Xc=Xd+BASELINE;Yc=Yd;Zc=Zd
uc=np.clip((FX_C*Xc/Zc+CX_C).astype(int),0,1919)
vc=np.clip((FY_C*Yc/Zc+CY_C).astype(int),0,1079)
plant_clr=rgb[vc,uc].astype(np.float64)/255.0

pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(plant_pts)
pcd.colors=o3d.utility.Vector3dVector(plant_clr)
pcd=pcd.voxel_down_sample(0.002)
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=15,std_ratio=3.0)
print(f"After cleanup: {len(pcd.points)} pts")

o3d.io.write_point_cloud(os.path.join(OUT,"plant_frame24.ply"),pcd)
print("Saved: plant_frame24.ply")

# Show plant alone
print("\nPLANT — close to continue.")
try:o3d.visualization.draw_geometries([pcd],window_name=f"Plant frame {FRAME} manual",width=1280,height=720)
except Exception as e:print(f"Viz:{e}")

# Show plant + pot together
pot=o3d.io.read_point_cloud(os.path.join(OUT,"pot_stage2.ply"))
both=pcd+pot
print(f"\nPLANT + POT ({len(both.points)} pts) — close window.")
try:o3d.visualization.draw_geometries([both],window_name="Pot + Plant frame 24",width=1280,height=720)
except Exception as e:print(f"Viz:{e}")
print("Done.")
