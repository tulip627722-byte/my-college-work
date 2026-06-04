"""2D annotation tool — draw polygon on RGB image to define crop region"""
import numpy as np
import cv2
import os
import json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Load frame 0 color image
c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
img_bgr = cv2.imdecode(c_raw, cv2.IMREAD_COLOR)
img_display = img_bgr.copy()

print(f"Image size: {img_bgr.shape[1]} x {img_bgr.shape[0]}")

# ---- Polygon state ----
polygon = []
drawing = img_display.copy()

def draw_poly(event, x, y, flags, param):
    global polygon, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon.append((x, y))
        cv2.circle(drawing, (x, y), 4, (0, 255, 0), -1)
        if len(polygon) > 1:
            cv2.line(drawing, polygon[-2], polygon[-1], (0, 255, 0), 2)
        cv2.imshow("Annotation", drawing)
        print(f"  Point {len(polygon)}: ({x}, {y})")

cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotation", 1280, 720)
cv2.setMouseCallback("Annotation", draw_poly)

print("=" * 50)
print("INSTRUCTIONS:")
print("  左键点击 = 添加多边形顶点")
print("  按 C     = 闭合多边形 + 确认")
print("  按 R     = 重置")
print("  按 ESC   = 退出不保存")
print("=" * 50)
print("\n每次点击添加一个顶点，沿花盆+绿植边界点一圈")
print("选完后按 C 闭合、保存mask并退出\n")

while True:
    cv2.imshow("Annotation", drawing)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        polygon = []
        drawing = img_display.copy()
        print("Reset.")

    elif key == ord('c'):
        if len(polygon) < 3:
            print("Need at least 3 points!")
            continue
        # Close polygon
        pts = np.array(polygon, dtype=np.int32)

        # Draw closed polygon
        cv2.polylines(drawing, [pts], True, (0, 255, 0), 2)
        cv2.imshow("Annotation", drawing)
        cv2.waitKey(500)

        # Create mask
        mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Save
        cv2.imwrite(os.path.join(OUT, "mask_2d.png"), mask)
        with open(os.path.join(OUT, "polygon_2d.json"), "w") as f:
            json.dump(polygon, f)
        print(f"\n✅ Mask saved: {OUT}/mask_2d.png")
        print(f"   Polygon: {polygon}")
        print(f"   Mask area: {np.sum(mask > 0)} pixels")
        break

    elif key == 27:  # ESC
        print("Cancelled.")
        break

cv2.destroyAllWindows()

# Also save visualization
if len(polygon) >= 3:
    pts = np.array(polygon, dtype=np.int32)
    overlay = img_bgr.copy()
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
    for i, pt in enumerate(polygon):
        cv2.circle(overlay, pt, 5, (0, 0, 255), -1)
        cv2.putText(overlay, str(i+1), (pt[0]+8, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT, "annotation_viz.png"), overlay)
    print(f"   Visualization: {OUT}/annotation_viz.png")
