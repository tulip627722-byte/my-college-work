"""
Rotation Axis Estimation — v2
==============================
Problem: Pot is geometrically symmetric → FGR fails on pot-only clouds.
Solution: Leverage 1920x1080 RGB texture (soil, pot markings) via SIFT matching.
         3D lift via depth → RANSAC rigid transform.

Strategy:
1. Load all 25 frames (depth + color)
2. For each adjacent pair:
   a. Extract pot/soil region in BOTH 3D and 2D (pixel coords)
   b. SIFT feature detection on the color ROI
   c. Match SIFT descriptors between adjacent frames
   d. Lift matched 2D points to 3D using depth map
   e. RANSAC rigid transform estimation
   f. ICP refinement on pot points
3. From transforms, solve for rotation axis
"""
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as Rsp
from scipy import io as sio
import pickle
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

from utils import (
    build_point_cloud, segment_pot, numpy_to_o3d, o3d_to_numpy,
    preprocess_pcd, compute_normals,
    FX_D, FY_D, CX_D, CY_D, DEPTH_H, DEPTH_W,
    FX_C, FY_C, CX_C, CY_C, COLOR_H, COLOR_W,
)

DATA_DIR = Path("E:/RGB-DEPTH/kinect_sync_data_manual")
OUTPUT_DIR = Path("E:/RGB-DEPTH/reconstruction")
N_FRAMES = 25


# ================================================================
#  RGB feature-based registration
# ================================================================

def load_frame(i):
    """Load frame i, return (depth_m, color_bgr, meta)."""
    depth_path = DATA_DIR / f"{i:04d}_depth.tif"
    color_path = DATA_DIR / f"{i:04d}_color.tif"

    if not depth_path.exists() or not color_path.exists():
        return None, None, None

    import tifffile
    depth_mm = tifffile.imread(str(depth_path)).astype(np.float64)
    depth_m = depth_mm / 1000.0
    color_bgr = tifffile.imread(str(color_path))
    if color_bgr.ndim == 3 and color_bgr.shape[2] == 3:
        color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)

    return depth_m, color_bgr, None


def get_pot_region_2d(depth_m, y_thresh_frac=0.55):
    """
    Identify pot/soil pixel region in 2D (depth image coordinates).
    Returns binary mask (424, 512) and list of (u,v) pixel coords.
    """
    h, w = depth_m.shape
    valid_depth = (depth_m > 0.3) & (depth_m < 2.5) & np.isfinite(depth_m)

    # Get 3D points for valid depth pixels
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth_m
    x = (uu - CX_D) * z / FX_D
    y = (vv - CY_D) * z / FY_D

    # Y threshold: pot is in lower portion
    y_median = np.median(y[valid_depth])
    y_low = (y < (y_median + 0.02)) & valid_depth

    # XZ radius filter: pot is centered, ~0.1m radius
    x_abs = np.abs(x)
    near_center = (x_abs < 0.15) & valid_depth

    mask = y_low & near_center
    return mask


def extract_sift_matches(img1_bgr, img2_bgr, mask1=None, mask2=None, ratio_thresh=0.75):
    """
    SIFT feature detection + matching between two color images.
    Optionally restrict to masked regions.
    Returns (kpts1, kpts2, good_matches).
    """
    sift = cv2.SIFT_create(nfeatures=2000)

    if mask1 is not None:
        mask1_u8 = mask1.astype(np.uint8) * 255
    else:
        mask1_u8 = None
    if mask2 is not None:
        mask2_u8 = mask2.astype(np.uint8) * 255
    else:
        mask2_u8 = None

    kp1, des1 = sift.detectAndCompute(img1_bgr, mask1_u8)
    kp2, des2 = sift.detectAndCompute(img2_bgr, mask2_u8)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return [], [], [], None, None

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    return kp1, kp2, good, des1, des2


def lift_2d_to_3d(kpts, depth_m, depth_shape=(424, 512)):
    """
    Lift 2D keypoints from COLOR image coordinates to 3D via depth map.
    Keypoints are in color image coords (1920x1080), need to map to depth coords (512x424).

    Color → Depth mapping: Kinect V2 has aligned frames, but different resolutions.
    We use the depth intrinsics directly: for each keypoint in color, we find
    the corresponding depth pixel and compute 3D.

    Simplified: we project keypoints from color coords to depth coords using
    known extrinsics, then back-project using depth intrinsics.
    """
    pts_3d = []
    valid_kpts = []

    # For each keypoint, find nearest depth pixel
    # Color → Depth mapping via Kinect SDK alignment
    # Approximate: scale color coords to depth resolution
    scale_u = DEPTH_W / COLOR_W  # 512/1920 ≈ 0.2667
    scale_v = DEPTH_H / COLOR_H  # 424/1080 ≈ 0.3926

    for kp in kpts:
        u_c = kp.pt[0]  # Color image X
        v_c = kp.pt[1]  # Color image Y

        # Map to depth pixel coords (rough alignment)
        u_d = int(u_c * scale_u)
        v_d = int(v_c * scale_v)

        # Check bounds
        if u_d < 0 or u_d >= DEPTH_W or v_d < 0 or v_d >= DEPTH_H:
            continue

        z_m = depth_m[v_d, u_d]
        if z_m <= 0.3 or z_m > 2.5 or not np.isfinite(z_m):
            continue

        # Back-project to 3D using depth intrinsics
        x = (u_d - CX_D) * z_m / FX_D
        y = (v_d - CY_D) * z_m / FY_D
        z = z_m

        pts_3d.append([x, y, z])
        valid_kpts.append(kp)

    return np.array(pts_3d), valid_kpts


def rgb_ransac_register(img1, depth1, img2, depth2, pot_mask1, pot_mask2):
    """
    Register frame 2 to frame 1 using SIFT + depth-lifted RANSAC.

    Returns:
        T_4x4: rigid transform mapping frame2 → frame1, or None
        n_inliers: number of RANSAC inliers
        rmse: inlier RMSE
    """
    # Detect and match SIFT features
    kp1, kp2, matches, des1, des2 = extract_sift_matches(
        img1, img2,
        mask1=pot_mask1.astype(np.uint8) if pot_mask1 is not None else None,
        mask2=pot_mask2.astype(np.uint8) if pot_mask2 is not None else None,
        ratio_thresh=0.75
    )

    if len(matches) < 20:
        return None, 0, float('inf')

    # Lift matched keypoints to 3D, keeping only pairs where BOTH have valid depth
    matched_pts_src = []  # frame 2 points (to be moved)
    matched_pts_dst = []  # frame 1 points (reference)

    scale_u = DEPTH_W / COLOR_W
    scale_v = DEPTH_H / COLOR_H

    for m in matches:
        # Source keypoint (frame 2)
        kp_s = kp2[m.trainIdx]
        u_s, v_s = int(kp_s.pt[0] * scale_u), int(kp_s.pt[1] * scale_v)
        if u_s < 0 or u_s >= DEPTH_W or v_s < 0 or v_s >= DEPTH_H:
            continue
        z_s = depth2[v_s, u_s]
        if z_s <= 0.3 or z_s > 2.5 or not np.isfinite(z_s):
            continue

        # Dest keypoint (frame 1)
        kp_d = kp1[m.queryIdx]
        u_d, v_d = int(kp_d.pt[0] * scale_u), int(kp_d.pt[1] * scale_v)
        if u_d < 0 or u_d >= DEPTH_W or v_d < 0 or v_d >= DEPTH_H:
            continue
        z_d = depth1[v_d, u_d]
        if z_d <= 0.3 or z_d > 2.5 or not np.isfinite(z_d):
            continue

        # Both valid — compute 3D
        x_s = (u_s - CX_D) * z_s / FX_D
        y_s = (v_s - CY_D) * z_s / FY_D
        x_d = (u_d - CX_D) * z_d / FX_D
        y_d = (v_d - CY_D) * z_d / FY_D

        matched_pts_src.append([x_s, y_s, z_s])
        matched_pts_dst.append([x_d, y_d, z_d])

    if len(matched_pts_src) < 10:
        return None, 0, float('inf')

    src_3d = np.array(matched_pts_src)
    dst_3d = np.array(matched_pts_dst)

    # RANSAC rigid transform (3D-3D)
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_3d)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst_3d)

    # Correspondence set (1-to-1)
    corr = o3d.utility.Vector2iVector(
        np.column_stack([np.arange(len(src_3d)), np.arange(len(src_3d))])
    )

    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_pcd, dst_pcd, corr,
            max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        return result.transformation, len(correspondence_set_from_result(result)), 0.0
    except Exception:
        # Fallback: manual RANSAC
        T, n_inliers = manual_ransac_rigid(src_3d, dst_3d, n_iter=5000, thresh=0.03)
        return T, n_inliers, 0.0 if T is not None else float('inf')


def correspondence_set_from_result(result):
    """Extract inlier set from RANSAC result."""
    try:
        return result.correspondence_set
    except AttributeError:
        return []


def manual_ransac_rigid(src, dst, n_iter=5000, thresh=0.03):
    """Manual RANSAC for rigid 3D-3D transform."""
    if len(src) < 3:
        return None, 0

    best_T = None
    best_inliers = 0
    n = len(src)

    for _ in range(n_iter):
        # Random 3 points
        idx = np.random.choice(n, 3, replace=False)
        T_i = estimate_rigid_3pt(src[idx], dst[idx])
        if T_i is None:
            continue

        # Count inliers
        src_h = np.column_stack([src, np.ones(n)])
        pred = (T_i @ src_h.T).T[:, :3]
        dist = np.linalg.norm(pred - dst, axis=1)
        inliers = np.sum(dist < thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_T = T_i

    return best_T, best_inliers


def estimate_rigid_3pt(src3, dst3):
    """Estimate rigid transform from 3 point correspondences."""
    # Compute centroids
    c_src = np.mean(src3, axis=0)
    c_dst = np.mean(dst3, axis=0)

    # Center
    H = (src3 - c_src).T @ (dst3 - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c_dst - R @ c_src

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ================================================================
#  Geometry-based registration (improved)
# ================================================================

def extract_soil_and_rim(xyz, rgb, depth_m_valid_mask=None):
    """
    Extract soil surface and pot rim — regions with texture and structure.
    Soil: Y is just below pot rim, dark brown color
    Rim: higher curvature points at pot boundary
    """
    # Soil: below median Y, non-green, with curvature
    y = xyz[:, 1]
    y_lo = np.percentile(y, 15)
    y_hi = np.percentile(y, 55)

    # Non-green (soil + pot)
    R, G, B = rgb[:, 2], rgb[:, 1], rgb[:, 0]
    exg = 2 * G - R - B
    non_leaf = exg < 0.05

    # Mid-height: between 15th and 55th percentile of Y
    mid_y = (y > y_lo) & (y < y_hi)

    mask = non_leaf & mid_y
    return mask


def icp_colored(source, target, init_transform=np.eye(4), voxel_size=0.003, max_dist=0.04):
    """Colored ICP refinement."""
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return init_transform, float('inf')

    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            src_down, tgt_down, max_dist, init_transform,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
            )
        )
        return result.transformation, result.inlier_rmse
    except Exception:
        return init_transform, float('inf')


# ================================================================
#  Rotation axis math
# ================================================================

def rotation_axis_angle_from_R(R):
    """Extract rotation axis direction and angle (degrees) from rotation matrix."""
    skew = R - R.T
    a = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(cos_theta)
    a_norm = np.linalg.norm(a)
    if a_norm < 1e-10:
        return np.array([0, 1, 0]), 0.0
    return a / a_norm, np.degrees(angle)


def solve_rotation_center(R_list, t_list):
    """Solve t = (I - R) * c for center c via least squares."""
    A = np.vstack([np.eye(3) - R for R in R_list])
    b = np.concatenate(t_list)
    c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return c, A @ c - b


def align_axis_signs(axes):
    """Align all axis vectors to same hemisphere."""
    ref = axes[0]
    return np.array([a if np.dot(a, ref) >= 0 else -a for a in axes])


# ================================================================
#  Main pipeline
# ================================================================

def main():
    print("=" * 60)
    print("ROTATION AXIS ESTIMATION v2 — SIFT + Depth RANSAC")
    print("=" * 60)

    # ---- Phase 1: Load all frames ----
    print("\n[Phase 1] Loading 25 frames...")
    frames = []
    for i in range(1, N_FRAMES + 1):
        depth_m, color_bgr, _ = load_frame(i)
        if depth_m is not None:
            pot_mask_2d = get_pot_region_2d(depth_m)
            frames.append({
                'id': i,
                'depth_m': depth_m,
                'color_bgr': color_bgr,
                'pot_mask_2d': pot_mask_2d,
            })
            n_pot = pot_mask_2d.sum()
            print(f"  Frame {i:02d}: depth={depth_m.shape}, pot_pixels={n_pot}")
    print(f"  Loaded {len(frames)} frames")

    # ---- Phase 2: Build full point clouds for later use ----
    print("\n[Phase 2] Building full point clouds...")
    all_pcds = []
    for fr in frames:
        pcd_dict = build_point_cloud(
            str(DATA_DIR / f"{fr['id']:04d}_depth.tif"),
            str(DATA_DIR / f"{fr['id']:04d}_color.tif"),
        )
        if pcd_dict and pcd_dict['n_kept'] > 1000:
            pcd = numpy_to_o3d(pcd_dict['xyz'], pcd_dict['rgb'])
            pcd = preprocess_pcd(pcd, voxel_size=0.002)
            all_pcds.append(pcd)
        else:
            all_pcds.append(None)
    print(f"  Built {sum(1 for p in all_pcds if p is not None)} point clouds")

    # ---- Phase 3: SIFT + RANSAC registration on each pair ----
    print("\n[Phase 3] SIFT + Depth RANSAC on adjacent pairs...")

    rotations = []
    translations = []
    angles = []
    axes = []
    rmses = []
    all_transforms = []

    for i in range(1, len(frames)):
        f_prev = frames[i - 1]
        f_curr = frames[i]

        print(f"\n  Pair {f_curr['id']:02d} -> {f_prev['id']:02d}:")
        sys.stdout.flush()

        # Method 1: SIFT on full color images (pot region only)
        T_sift, n_inliers, _ = rgb_ransac_register(
            f_prev['color_bgr'], f_prev['depth_m'],
            f_curr['color_bgr'], f_curr['depth_m'],
            f_prev['pot_mask_2d'], f_curr['pot_mask_2d']
        )

        T_best = None
        method_used = "none"

        if T_sift is not None and n_inliers >= 15:
            R_sift, t_sift = T_sift[:3, :3], T_sift[:3, 3]
            axis_sift, angle_sift = rotation_axis_angle_from_R(R_sift)
            T_best = T_sift
            method_used = f"SIFT ({n_inliers} inliers)"
            print(f"    {method_used}: angle={angle_sift:.2f} deg")
        else:
            print(f"    SIFT: FAILED ({n_inliers} inliers)")

        # Method 2 fallback: ICP between full point clouds using identity init
        if T_best is None and all_pcds[f_curr['id'] - 1] is not None and all_pcds[f_prev['id'] - 1] is not None:
            pcd_curr = all_pcds[f_curr['id'] - 1]
            pcd_prev = all_pcds[f_prev['id'] - 1]

            T_icp, rmse = icp_colored(pcd_curr, pcd_prev, np.eye(4))
            R_icp, t_icp = T_icp[:3, :3], T_icp[:3, 3]
            axis_icp, angle_icp = rotation_axis_angle_from_R(R_icp)

            if rmse < 0.05:
                T_best = T_icp
                method_used = f"ColoredICP (rmse={rmse:.4f})"
                print(f"    {method_used}: angle={angle_icp:.2f} deg")
            else:
                print(f"    ColoredICP: rmse={rmse:.4f} too high, SKIPPED")

        if T_best is None:
            print("    -> PAIR FAILED, using identity")
            T_best = np.eye(4)

        # Store results
        R, t = T_best[:3, :3], T_best[:3, 3]
        axis, angle = rotation_axis_angle_from_R(R)

        rotations.append(R)
        translations.append(t)
        angles.append(angle)
        axes.append(axis)
        all_transforms.append(T_best)

    # ---- Phase 4: Estimate rotation axis from transforms ----
    print("\n" + "=" * 60)
    print("[Phase 4] Estimating rotation axis from transforms...")

    if len(rotations) < 5:
        print("FATAL: Too few transforms to estimate axis")
        return

    # Robust filtering: remove obvious outliers
    expected_angle = 360.0 / N_FRAMES  # 14.4 deg
    good_pairs = []
    for j, angle in enumerate(angles):
        angle_error = abs(angle - expected_angle)
        if angle_error < 20:  # within 20 deg of expected
            good_pairs.append(j)
            print(f"  Pair {j+2:02d}->{j+1:02d}: angle={angle:.2f} deg (error={angle_error:.1f}) KEPT")
        else:
            print(f"  Pair {j+2:02d}->{j+1:02d}: angle={angle:.2f} deg (error={angle_error:.1f}) OUTLIER")

    print(f"\n  Keeping {len(good_pairs)}/{len(angles)} pairs for axis estimation")

    if len(good_pairs) < 5:
        print("FATAL: Too few good pairs")
        return

    # Filter to good pairs
    R_good = [rotations[j] for j in good_pairs]
    t_good = [translations[j] for j in good_pairs]
    angles_good = [angles[j] for j in good_pairs]
    axes_good = [axes[j] for j in good_pairs]

    # Estimate axis direction
    axes_aligned = align_axis_signs(axes_good)
    median_axis = np.median(axes_aligned, axis=0)
    median_axis = median_axis / np.linalg.norm(median_axis)
    median_angle = np.median(angles_good)

    # Estimate center
    center, residuals = solve_rotation_center(R_good, t_good)

    # Per-pair center std
    per_pair_centers = []
    for R, t in zip(R_good, t_good):
        try:
            ci = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]
            per_pair_centers.append(ci)
        except np.linalg.LinAlgError:
            pass

    axis_dispersion = np.degrees(np.mean([
        np.arccos(np.clip(np.abs(np.dot(a, median_axis)), 0, 1))
        for a in axes_good
    ]))

    print(f"\n  === AXIS ESTIMATE ===")
    print(f"  Direction:   [{median_axis[0]:.4f}, {median_axis[1]:.4f}, {median_axis[2]:.4f}]")
    print(f"  Center:      [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")
    print(f"  Angle/step:  {median_angle:.2f} deg (expected {expected_angle:.1f})")
    print(f"  Axis dispersion: {axis_dispersion:.2f} deg")
    print(f"  Angle std:   {np.std(angles_good):.2f} deg")
    if len(per_pair_centers) > 0:
        center_std = np.std(np.array(per_pair_centers), axis=0)
        print(f"  Center std:  [{center_std[0]:.4f}, {center_std[1]:.4f}, {center_std[2]:.4f}] m")

    # ---- Save results ----
    axis_info = {
        'axis': median_axis,
        'angle_per_step': median_angle,
        'center': center,
        'per_pair_angles': angles,
        'per_pair_axes': axes_aligned,
        'good_pairs': good_pairs,
        'n_good': len(good_pairs),
        'n_total_pairs': len(angles),
        'axis_dispersion': axis_dispersion,
        'angle_expected': expected_angle,
    }

    with open(OUTPUT_DIR / "axis_estimate_v2.pkl", 'wb') as f:
        pickle.dump(axis_info, f)

    np.savez(
        OUTPUT_DIR / "axis_estimate_v2.npz",
        axis=median_axis,
        center=center,
        angle_per_step=median_angle,
        angles=np.array(angles),
        axes=np.array(axes_aligned),
        n_good=len(good_pairs),
        good_pairs=np.array(good_pairs),
    )

    print(f"\n  Results saved to {OUTPUT_DIR / 'axis_estimate_v2.*'}")

    # ---- Phase 5: Quick visual check ----
    print("\n[Phase 5] Quick validation...")
    print(f"  Axis is mostly Y? {abs(median_axis[1]) > 0.5}: [{median_axis[0]:.3f}, {median_axis[1]:.3f}, {median_axis[2]:.3f}]")
    print(f"  Angle matches 14.4? {abs(median_angle - expected_angle) < 5}: {median_angle:.1f} deg")
    print(f"  Center near origin? r={np.linalg.norm(center):.4f} m")
    print(f"  Good pairs / total: {len(good_pairs)}/{len(angles)}")

    quality_score = 0
    if abs(median_angle - expected_angle) < 5:
        quality_score += 1
    if axis_dispersion < 10:
        quality_score += 1
    if len(good_pairs) >= 15:
        quality_score += 1

    if quality_score >= 3:
        print("\n  *** AXIS ESTIMATE LOOKS RELIABLE ***")
    elif quality_score >= 2:
        print("\n  ** Axis estimate may be usable with caution **")
    else:
        print("\n  * Axis estimate UNRELIABLE — consider other approaches *")

    return axis_info, frames, all_transforms


if __name__ == "__main__":
    main()
