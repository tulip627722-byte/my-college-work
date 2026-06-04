"""Rectangle-select pot/plant color regions on RGB image"""
import numpy as np
import cv2
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
img_bgr = cv2.imdecode(c_raw, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h_img, w_img = img_rgb.shape[:2]

# State
mode = 'pot'  # 'pot' | 'plant'
pot_rects = []    # [(x1,y1,x2,y2), ...]
plant_rects = []
drawing = False
start_pt = None
end_pt = None

def draw_ui():
    d = img_rgb.copy()
    # Draw rectangles
    for x1, y1, x2, y2 in pot_rects:
        cv2.rectangle(d, (x1, y1), (x2, y2), (255, 50, 50), 2)
    for x1, y1, x2, y2 in plant_rects:
        cv2.rectangle(d, (x1, y1), (x2, y2), (50, 255, 50), 2)
    # Current drag preview
    if drawing and start_pt and end_pt:
        c = (255, 50, 50) if mode == 'pot' else (50, 255, 50)
        cv2.rectangle(d, start_pt, end_pt, c, 2)
    # Status bar
    cv2.rectangle(d, (0, 0), (w_img, 42), (30, 30, 30), -1)
    status = f"MODE: {mode.upper()}  |  Pot(red):{len(pot_rects)} rects  Plant(green):{len(plant_rects)} rects  |  SPACE=toggle  C=confirm  R=reset"
    cv2.putText(d, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return d

def on_mouse(event, x, y, flags, param):
    global drawing, start_pt, end_pt
    if y < 42:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_pt = None
        x1, y1 = start_pt
        x2, y2 = x, y
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 > 5 and y2 - y1 > 5:
            if mode == 'pot':
                pot_rects.append((x1, y1, x2, y2))
            else:
                plant_rects.append((x1, y1, x2, y2))

cv2.namedWindow("Color Sampler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Sampler", 1280, 720)
cv2.setMouseCallback("Color Sampler", on_mouse)

print("=" * 55)
print("  POT mode (red rects)  - drag to select pot body")
print("  PLANT mode (green rects) - drag to select leaves")
print()
print("  SPACE = toggle mode    R = reset    C = confirm")
print("=" * 55)

while True:
    ui = draw_ui()
    cv2.imshow("Color Sampler", cv2.cvtColor(ui, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(30) & 0xFF

    if key == ord(' '):
        mode = 'plant' if mode == 'pot' else 'pot'
        print(f"Mode: {mode.upper()}")

    elif key == ord('r'):
        pot_rects.clear()
        plant_rects.clear()
        print("Reset all rectangles.")

    elif key == ord('c'):
        if not pot_rects or not plant_rects:
            print("Need at least 1 rectangle for each!")
            continue
        break

    elif key == 27:
        cv2.destroyAllWindows()
        exit("Cancelled.")

cv2.destroyAllWindows()

# Collect color samples from rectangles
def collect_colors(rects):
    samples = []
    for x1, y1, x2, y2 in rects:
        roi = img_rgb[y1:y2, x1:x2]
        samples.append(roi.reshape(-1, 3))
    if samples:
        return np.vstack(samples)
    return np.array([])

pot_colors = collect_colors(pot_rects)
plant_colors = collect_colors(plant_rects)

print(f"\nPot samples: {len(pot_colors)} pixels, mean RGB=({pot_colors[:,0].mean():.0f},{pot_colors[:,1].mean():.0f},{pot_colors[:,2].mean():.0f})")
print(f"Plant samples: {len(plant_colors)} pixels, mean RGB=({plant_colors[:,0].mean():.0f},{plant_colors[:,1].mean():.0f},{plant_colors[:,2].mean():.0f})")

# Save
np.savez(os.path.join(OUT, "color_model.npz"),
         pot_mean=pot_colors.mean(axis=0), pot_std=pot_colors.std(axis=0),
         plant_mean=plant_colors.mean(axis=0), plant_std=plant_colors.std(axis=0))
print("Saved color_model.npz")

# Test on frame 0
import open3d as o3d
pcd = o3d.io.read_point_cloud(os.path.join(DATA, "output", "cropped_frames", "frame_000_crop.ply"))
pts = np.asarray(pcd.points)
colors_01 = np.asarray(pcd.colors) * 255

pot_mean = pot_colors.mean(axis=0)
plant_mean = plant_colors.mean(axis=0)
dist_pot = np.sqrt(((colors_01 - pot_mean)**2).sum(axis=1))
dist_plant = np.sqrt(((colors_01 - plant_mean)**2).sum(axis=1))
is_pot = dist_pot < dist_plant

print(f"\nFrame 0 classification:")
print(f"  Pot:   {np.sum(is_pot)} pts  Y=[{pts[is_pot,1].min():.3f},{pts[is_pot,1].max():.3f}]")
print(f"  Plant: {np.sum(~is_pot)} pts  Y=[{pts[~is_pot,1].min():.3f},{pts[~is_pot,1].max():.3f}]")

pcd_viz = o3d.geometry.PointCloud()
pcd_viz.points = o3d.utility.Vector3dVector(pts)
clr_viz = np.zeros((len(pts), 3))
clr_viz[is_pot] = [0.3, 0.3, 0.8]
clr_viz[~is_pot] = [0.2, 0.8, 0.2]
pcd_viz.colors = o3d.utility.Vector3dVector(clr_viz)
o3d.io.write_point_cloud(os.path.join(OUT, "debug_frame000_manual_sep.ply"), pcd_viz)
print("Saved debug_frame000_manual_sep.ply — verify then I batch all 36.")
