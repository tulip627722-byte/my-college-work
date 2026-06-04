"""Step 4 v6: ECC image alignment on RGB ROI to estimate rotation"""
import numpy as np
import cv2
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Load images and crop to ROI
mask_raw = np.fromfile(os.path.join(OUT, "mask_2d.png"), dtype=np.uint8)
mask = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
mask_dilated = cv2.dilate(mask, kernel, iterations=2)
ys, xs = np.where(mask_dilated > 0)
x1, x2 = xs.min(), xs.max()
y1, y2 = ys.min(), ys.max()
print(f"ROI: {x2-x1}x{y2-y1}")

imgs_gray = []
for fidx in range(36):
    raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs_gray.append(gray[y1:y2, x1:x2])

# ECC alignment
warp_mode = cv2.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

angles = [0.0]
cumulative = 0.0

for fidx in range(1, 36):
    prev = imgs_gray[fidx-1]
    curr = imgs_gray[fidx]

    try:
        _, warp_matrix = cv2.findTransformECC(
            prev, curr, warp_matrix, warp_mode, criteria, None, 5)

        # Extract rotation from Euclidean warp matrix
        # warp = [[cos, -sin, tx], [sin, cos, ty]]
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)

        cumulative += angle_deg
        angles.append(cumulative)
        print(f"Frame {fidx:03d}: step={angle_deg:+.3f} deg  total={cumulative:.2f} deg  tx={warp_matrix[0,2]:.2f} ty={warp_matrix[1,2]:.2f}")
    except Exception as e:
        print(f"Frame {fidx:03d}: ECC failed ({e})")
        angles.append(cumulative)
        warp_matrix = np.eye(2, 3, dtype=np.float32)  # reset

angles_arr = np.array(angles)
diffs = np.diff(angles_arr)
print(f"\n{'='*60}")
print(f"ECC Results:")
print(f"  Diffs: {[f'{d:.2f}' for d in diffs]}")
print(f"  Same sign? {'YES' if np.all(diffs>0) or np.all(diffs<0) else 'NO'}  (+{np.sum(diffs>0)}/-{np.sum(diffs<0)})")
print(f"  Median |step|: {np.median(np.abs(diffs)):.3f} deg/frame")
print(f"  Total: {angles_arr[-1] - angles_arr[0]:.2f} deg")
print(f"  Range: {angles_arr[0]:.1f} -> {angles_arr[-1]:.1f} deg")

pose = {
    "method": "ECC image alignment",
    "angles_per_frame_deg": [float(a) for a in angles_arr],
    "median_step_deg": float(np.median(np.abs(diffs))),
    "total_rotation_deg": float(angles_arr[-1] - angles_arr[0]),
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose, f, indent=2)
print(f"\nSaved phase_angles.json")
