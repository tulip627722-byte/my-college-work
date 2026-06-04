"""
Manual annotation tool for frame 24 plant (fixed Chinese path)
Left click = add point, Right click = close polygon, C = confirm & save
"""
import numpy as np, cv2, os

DATA=r"E:\花瓶3d"
raw=np.fromfile(os.path.join(DATA,"color","024.png"),dtype=np.uint8)
img=cv2.imdecode(raw,cv2.IMREAD_COLOR)
h,w=img.shape[:2]

scale=0.5
disp=cv2.resize(img,(int(w*scale),int(h*scale)))
points=[]

def draw_poly(event,x,y,flags,param):
    global points
    if event==cv2.EVENT_LBUTTONDOWN:
        points.append((int(x/scale),int(y/scale)))
        cv2.circle(disp,(x,y),3,(0,255,0),-1)
        if len(points)>1:
            pts=np.array([[int(p[0]*scale),int(p[1]*scale)] for p in points])
            cv2.polylines(disp,[pts],False,(0,255,0),2)
        cv2.imshow("Frame 24 - draw plant mask",disp)
    elif event==cv2.EVENT_RBUTTONDOWN:
        if len(points)>2:
            pts=np.array([[int(p[0]*scale),int(p[1]*scale)] for p in points])
            cv2.polylines(disp,[pts],True,(0,255,0),2)
            cv2.imshow("Frame 24 - draw plant mask",disp)

print("Draw polygon around plant (left=add, right=close, C=save, R=reset, Q=quit)")
cv2.imshow("Frame 24 - draw plant mask",disp)
cv2.setMouseCallback("Frame 24 - draw plant mask",draw_poly)

while True:
    k=cv2.waitKey(0)&0xFF
    if k==ord('c') or k==ord('C'):
        if len(points)<3:
            print("Need at least 3 points!")
            continue
        mask=np.zeros((h,w),dtype=np.uint8)
        pts=np.array([points],dtype=np.int32)
        cv2.fillPoly(mask,pts,255)
        # Chinese path workaround
        out_path=os.path.join(DATA,"output","mask_plant_frame24.png")
        ok,enc=cv2.imencode('.png',mask)
        if ok:
            with open(out_path,'wb') as f:f.write(enc.tobytes())
            print(f"Saved! ({len(points)} points, {mask.sum()//255} white pixels)")
        else:
            print("Encode failed!")
        # Show overlay
        overlay=img.copy()
        overlay[mask>0]=cv2.addWeighted(overlay[mask>0],0.5,np.full_like(overlay[mask>0],(0,200,0)),0.5,0)
        cv2.imshow("Saved mask",cv2.resize(overlay,(int(w*scale),int(h*scale))))
        cv2.waitKey(1000)
        break
    elif k==ord('r') or k==ord('R'):
        points=[]
        disp=cv2.resize(img,(int(w*scale),int(h*scale)))
        cv2.imshow("Frame 24 - draw plant mask",disp)
        print("Reset")
    elif k==ord('q') or k==ord('Q'):
        print("Quit without saving")
        break

cv2.destroyAllWindows()
print("Done.")
