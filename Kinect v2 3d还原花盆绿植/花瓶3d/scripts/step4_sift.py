"""Step 4 v5: SIFT feature matching on RGB images to estimate rotation"""
import numpy as np
import cv2
import os, json

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

# Load all color images
imgs = []
for fidx in range(36):
    raw = np.fromfile(os.path.join(DATA, "color", f"{fidx:03d}.png"), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    imgs.append(img)

# Crop to ROI (same as mask region)
# Load mask to define ROI
mask_raw = np.fromfile(os.path.join(OUT, "mask_2d.png"), dtype=np.uint8)
mask = cv2.imdecode(mask_raw, cv2.IMREAD_GRAYSCALE)
# Dilate mask generously
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
mask_dilated = cv2.dilate(mask, kernel, iterations=2)
ys, xs = np.where(mask_dilated > 0)
if len(xs) > 0:
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
else:
    x1, y1, x2, y2 = 700, 400, 1100, 850

print(f"ROI: x=[{x1},{x2}] y=[{y1},{y2}] ({x2-x1}x{y2-y1})")

sift = cv2.SIFT_create(nfeatures=500)

# For each adjacent pair, match features and estimate rotation
angles = [0.0]
cumulative = 0.0

for fidx in range(1, 36):
    img_prev = imgs[fidx-1][y1:y2, x1:x2]
    img_curr = imgs[fidx][y1:y2, x1:x2]

    # Convert to grayscale
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray_prev, None)
    kp2, des2 = sift.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        print(f"Frame {fidx:03d}: insufficient features ({len(des1) if des1 is not None else 0}, {len(des2) if des2 is not None else 0})")
        angles.append(cumulative)
        continue

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"Frame {fidx:03d}: {len(kp1)}/{len(kp2)} keypoints, {len(good)} good matches", end="")

    if len(good) < 8:
        print(" -> insufficient matches")
        angles.append(cumulative)
        continue

    # Estimate affine transform
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    # Use findHomography or estimateAffinePartial2D
    try:
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            print(" -> affine failed")
            angles.append(cumulative)
            continue

        # Extract rotation from 2x3 affine matrix
        # M = [[a, b, tx], [c, d, ty]]
        # Rotation angle = atan2(c, a) or atan2(-b, a)
        angle_rad = np.arctan2(M[0, 1], M[0, 0])
        angle_deg = np.degrees(angle_rad)

        # The 2D image rotation is roughly related to the 3D Y-axis rotation
        # For a turntable, the image rotation ≈ the turntable rotation
        cumulative += angle_deg
        angles.append(cumulative)

        inlier_count = np.sum(inliers) if inliers is not None else 0
        print(f" -> angle={angle_deg:+.2f} deg  total={cumulative:.1f} deg  inliers={inlier_count}/{len(good)}")
    except Exception as e:
        print(f" -> error: {e}")
        angles.append(cumulative)

angles_arr = np.array(angles)
diffs = np.diff(angles_arr)
print(f"\n{'='*60}")
print(f"SIFT Phase Angle Results:")
print(f"  Diffs: {[f'{d:.1f}' for d in diffs]}")
print(f"  Same sign? {'YES' if np.all(diffs>0) or np.all(diffs<0) else 'NO'}: {np.sum(diffs>0)}/+  {np.sum(diffs<0)}/-")
print(f"  Median step: {np.median(np.abs(diffs)):.2f} deg/frame")
print(f"  Total rotation: {abs(angles_arr[-1] - angles_arr[0]):.1f} deg")

# Save
pose = {
    "method": "SIFT feature matching",
    "angles_per_frame_deg": [float(a) for a in angles_arr],
    "median_step_deg": float(np.median(np.abs(diffs))),
    "total_rotation_deg": float(abs(angles_arr[-1] - angles_arr[0])),
}
with open(os.path.join(OUT, "phase_angles.json"), "w") as f:
    json.dump(pose, f, indent=2)
print(f"\nSaved phase_angles.json")
