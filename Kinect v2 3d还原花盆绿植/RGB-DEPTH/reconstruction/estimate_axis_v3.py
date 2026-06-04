"""
Rotation Axis Estimation — v3 (Direct Colored ICP)
====================================================
Key insight: Skip feature matching entirely.
Use Colored ICP on the soil surface region directly.
Soil has RGB texture → Colored ICP uses joint color+geometry.
If it converges to ~14.4° rotation, we know the axis.

If identity→ICP doesn't converge (14.4° too big), we brute-force
try multiple initial rotation axes and pick the best.
"""
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as Rsp
import pickle
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')

from utils import (
    build_point_cloud, numpy_to_o3d, preprocess_pcd,
    FX_D, FY_D, CX_D, CY_D, DEPTH_H, DEPTH_W,
)

DATA_DIR = Path("E:/RGB-DEPTH/kinect_sync_data_manual")
OUTPUT_DIR = Path("E:/RGB-DEPTH/reconstruction")
N_FRAMES = 25


def load_frame_raw(i):
    """Load just depth + color arrays."""
    import tifffile
    depth_mm = tifffile.imread(str(DATA_DIR / f"{i:04d}_depth.tif")).astype(np.float64)
    depth_m = depth_mm / 1000.0
    color_bgr = tifffile.imread(str(DATA_DIR / f"{i:04d}_color.tif"))
    if color_bgr.ndim == 3 and color_bgr.shape[2] == 3:
        color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)
    return depth_m, color_bgr


def extract_soil_region(xyz, rgb, depth_m, color_bgr):
    """
    Extract soil + pot rim — rigid textured region.
    Uses relaxed thresholds based on data analysis:
    - Y range: lower portion of plant (pot is at bottom)
    - Non-green color filter
    - Z range: keep main object, exclude far background
    """
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Pot is at the bottom of the point cloud
    # Use robust percentiles computed on ALL valid points
    y_lo = np.percentile(y, 2)    # very bottom
    y_hi = np.percentile(y, 55)   # up to middle of plant

    # Non-green: soil/pot are brown, not green
    R = rgb[:, 2].astype(np.float64)
    G = rgb[:, 1].astype(np.float64)
    B = rgb[:, 0].astype(np.float64)
    exg = 2 * G - R - B
    is_not_leaf = exg < 0.08  # relaxed: catch brown/dark green

    # Y range
    in_y = (y >= y_lo) & (y <= y_hi)

    # Z depth: main object depth (based on actual data: Z ≈ 0.5-2.1m)
    z_median = np.median(z)
    in_depth = (z > 0.4) & (z < z_median + 0.3)

    mask = is_not_leaf & in_y & in_depth

    return xyz[mask], rgb[mask], mask


def colored_icp_register(source_xyz, source_rgb, target_xyz, target_rgb,
                         init_transform=np.eye(4),
                         voxel_size=0.003, max_dist=0.05):
    """
    Colored ICP registration using joint color + geometry.
    Returns (transform_4x4, rmse, fitness).
    """
    src = numpy_to_o3d(source_xyz, source_rgb)
    tgt = numpy_to_o3d(target_xyz, target_rgb)

    src = preprocess_pcd(src, voxel_size=voxel_size, nb_neighbors=10, std_ratio=3.0)
    tgt = preprocess_pcd(tgt, voxel_size=voxel_size, nb_neighbors=10, std_ratio=3.0)

    if len(src.points) < 50 or len(tgt.points) < 50:
        return None, float('inf'), 0.0

    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))

    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            src, tgt, max_dist, init_transform,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=0.7  # weight geometry more for robustness
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=150
            )
        )
        return result.transformation, result.inlier_rmse, result.fitness
    except Exception as e:
        return None, float('inf'), 0.0


def rotation_axis_angle_from_R(R):
    """Extract axis and angle (degrees) from rotation matrix."""
    skew = R - R.T
    a = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(cos_theta)
    a_norm = np.linalg.norm(a)
    if a_norm < 1e-10:
        return np.array([0, 1, 0]), 0.0
    return a / a_norm, np.degrees(angle)


def solve_rotation_center(R_list, t_list):
    """Solve t = (I-R)*c for center c."""
    A = np.vstack([np.eye(3) - R for R in R_list])
    b = np.concatenate(t_list)
    c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return c, A @ c - b


def brute_force_axis_search(source_xyz, source_rgb, target_xyz, target_rgb,
                            n_samples=36):
    """
    Try many initial rotation axes, run Colored ICP from each,
    return the best transform.
    """
    best_transform = None
    best_score = float('inf')
    best_result = None
    angle = 14.4  # expected per-step rotation

    # Sample axes uniformly on sphere
    results = []
    for i in range(n_samples):
        # Fibonacci sphere sampling
        phi = np.arccos(1 - 2 * (i + 0.5) / n_samples)
        theta = np.pi * (1 + np.sqrt(5)) * i
        ax = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])

        # Normalize
        ax = ax / np.linalg.norm(ax)

        # Build initial transform: rotation about ax by 14.4°
        R_init = Rsp.from_rotvec(ax * np.radians(angle)).as_matrix()
        T_init = np.eye(4)
        T_init[:3, :3] = R_init

        T, rmse, fitness = colored_icp_register(
            source_xyz, source_rgb, target_xyz, target_rgb, T_init,
            voxel_size=0.004, max_dist=0.04
        )

        if T is not None and rmse < best_score:
            best_score = rmse
            best_transform = T
            best_result = (ax, rmse, fitness)

        results.append({'axis': ax.tolist(), 'rmse': float(rmse), 'fitness': float(fitness)})

    return best_transform, best_score, best_result, results


def main():
    print("=" * 60)
    print("ROTATION AXIS ESTIMATION v3 — Direct Colored ICP on Soil")
    print("=" * 60)

    # ---- Phase 1: Load, build point clouds, extract soil ----
    print("\n[Phase 1] Loading frames + extracting soil regions...")

    soil_clouds = []  # (xyz, rgb) for soil region of each frame
    full_pcds = []

    for i in range(1, N_FRAMES + 1):
        depth_m, color_bgr = load_frame_raw(i)
        pcd_dict = build_point_cloud(
            str(DATA_DIR / f"{i:04d}_depth.tif"),
            str(DATA_DIR / f"{i:04d}_color.tif"),
        )

        if pcd_dict is None or pcd_dict['n_kept'] < 1000:
            print(f"  Frame {i:02d}: FAILED")
            soil_clouds.append(None)
            full_pcds.append(None)
            continue

        # Extract soil
        soil_xyz, soil_rgb, soil_mask = extract_soil_region(
            pcd_dict['xyz'], pcd_dict['rgb'], depth_m, color_bgr
        )

        full_pcds.append(numpy_to_o3d(pcd_dict['xyz'], pcd_dict['rgb']))
        soil_clouds.append((soil_xyz, soil_rgb))

        print(f"  Frame {i:02d}: full={len(pcd_dict['xyz'])}, soil={len(soil_xyz)}")

    # ---- Phase 2: Try basic approach first — Colored ICP from identity ----
    print("\n[Phase 2] Colored ICP on soil, identity init...")

    # First, try identity on the first pair to see if Colored ICP handles 14.4°
    if soil_clouds[0] is not None and soil_clouds[1] is not None:
        s_xyz, s_rgb = soil_clouds[1]  # source (frame 2)
        t_xyz, t_rgb = soil_clouds[0]  # target (frame 1)

        T_ident, rmse, fitness = colored_icp_register(
            s_xyz, s_rgb, t_xyz, t_rgb, np.eye(4),
            voxel_size=0.003, max_dist=0.06
        )

        print(f"\n  Pair 02->01, identity init:")
        if T_ident is not None:
            R_ident = T_ident[:3, :3]
            axis_ident, angle_ident = rotation_axis_angle_from_R(R_ident)
            print(f"    RMSE={rmse:.5f}m, fitness={fitness:.3f}, angle={angle_ident:.2f} deg")
            print(f"    Axis=[{axis_ident[0]:.3f}, {axis_ident[1]:.3f}, {axis_ident[2]:.3f}]")

            if abs(angle_ident - 14.4) < 5:
                print("    *** Colored ICP converged to ~14.4 deg from identity! ***")
            else:
                print("    Colored ICP did NOT find the correct rotation from identity.")
                print("    -> Will try brute-force axis search...")
        else:
            print("    FAILED")
    else:
        print("  First pair unavailable")

    # ---- Phase 3: Brute-force axis search on first pair (if needed) ----
    print("\n[Phase 3] Brute-force axis search on pair 02->01...")

    if soil_clouds[0] is not None and soil_clouds[1] is not None:
        s_xyz, s_rgb = soil_clouds[1]
        t_xyz, t_rgb = soil_clouds[0]

        best_T, best_score, best_result, all_results = brute_force_axis_search(
            s_xyz, s_rgb, t_xyz, t_rgb, n_samples=72
        )

        if best_T is not None:
            R_best = best_T[:3, :3]
            axis_best, angle_best = rotation_axis_angle_from_R(R_best)
            init_axis, init_rmse, init_fitness = best_result
            print(f"\n  Best result:")
            print(f"    Init axis:  [{init_axis[0]:.3f}, {init_axis[1]:.3f}, {init_axis[2]:.3f}]")
            print(f"    Final axis: [{axis_best[0]:.3f}, {axis_best[1]:.3f}, {axis_best[2]:.3f}]")
            print(f"    Final angle: {angle_best:.2f} deg")
            print(f"    RMSE: {init_rmse:.5f}m, fitness: {init_fitness:.3f}")

            # Find the best axis by RMSE
            sorted_results = sorted(all_results, key=lambda r: r['rmse'])
            print(f"\n  Top 5 axis hypotheses:")
            for j, r in enumerate(sorted_results[:5]):
                a = r['axis']
                print(f"    {j+1}. axis=[{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}], rmse={r['rmse']:.5f}, fit={r['fitness']:.3f}")
    else:
        print("  Skipped (no data)")

    # ---- Phase 4: Register all pairs with best axis init ----
    print("\n[Phase 4] Register all 24 pairs with best axis from brute-force...")

    # Get the best init axis from phase 3
    best_init_axis = None
    if 'sorted_results' in dir() and len(sorted_results) > 0:
        best_init_axis = np.array(sorted_results[0]['axis'])
    else:
        best_init_axis = np.array([0, 1, 0])  # fallback: Y axis

    expected_angle = 360.0 / N_FRAMES

    rotations = []
    translations = []
    angles = []
    axes = []
    rmses_list = []
    pair_results = []

    for i in range(1, N_FRAMES):
        if soil_clouds[i] is None or soil_clouds[i-1] is None:
            print(f"  Pair {i+1:02d}->{i:02d}: SKIPPED (no data)")
            continue

        s_xyz, s_rgb = soil_clouds[i]    # source
        t_xyz, t_rgb = soil_clouds[i-1]  # target

        # Initial transform using best axis
        R_init = Rsp.from_rotvec(best_init_axis * np.radians(expected_angle)).as_matrix()
        T_init = np.eye(4)
        T_init[:3, :3] = R_init

        T, rmse, fitness = colored_icp_register(
            s_xyz, s_rgb, t_xyz, t_rgb, T_init,
            voxel_size=0.003, max_dist=0.04
        )

        if T is not None:
            R = T[:3, :3]
            t = T[:3, 3]
            axis, angle = rotation_axis_angle_from_R(R)

            rotations.append(R)
            translations.append(t)
            angles.append(angle)
            axes.append(axis)
            rmses_list.append(rmse)
            pair_results.append((i, angle, rmse, fitness, axis))

            status = "OK" if abs(angle - expected_angle) < 8 else f"ANGLE_OFF({angle:.1f})"
            print(f"  Pair {i+1:02d}->{i:02d}: angle={angle:.2f}, rmse={rmse:.4f}, fit={fitness:.3f} [{status}]")
        else:
            print(f"  Pair {i+1:02d}->{i:02d}: ICP FAILED")

    # ---- Phase 5: Solve for axis from all transforms ----
    print("\n" + "=" * 60)
    print("[Phase 5] Solving for rotation axis...")

    if len(rotations) < 5:
        print(f"  FATAL: only {len(rotations)} successful pairs")
        return

    # Filter outliers
    good_idx = []
    for j, angle in enumerate(angles):
        if abs(angle - expected_angle) < 15:
            good_idx.append(j)

    R_good = [rotations[j] for j in good_idx]
    t_good = [translations[j] for j in good_idx]
    angles_good = [angles[j] for j in good_idx]
    axes_good = [axes[j] for j in good_idx]

    # Align signs
    ref = axes_good[0]
    axes_good_aligned = []
    for a in axes_good:
        if np.dot(a, ref) < 0:
            axes_good_aligned.append(-a)
        else:
            axes_good_aligned.append(a)
    axes_good_aligned = np.array(axes_good_aligned)

    median_axis = np.median(axes_good_aligned, axis=0)
    median_axis = median_axis / np.linalg.norm(median_axis)
    median_angle = np.median(angles_good)

    center, residuals = solve_rotation_center(R_good, t_good)

    axis_dispersion = np.degrees(np.mean([
        np.arccos(np.clip(np.abs(np.dot(a, median_axis)), 0, 1))
        for a in axes_good_aligned
    ]))

    print(f"\n  Good pairs: {len(good_idx)}/{len(angles)}")
    print(f"  Axis direction:  [{median_axis[0]:.4f}, {median_axis[1]:.4f}, {median_axis[2]:.4f}]")
    print(f"  Rotation center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  Angle per step:  {median_angle:.2f} deg")
    print(f"  Axis dispersion: {axis_dispersion:.2f} deg")
    print(f"  Angle std:       {np.std(angles_good):.2f} deg")

    # ---- Save ----
    results = {
        'axis': median_axis.tolist(),
        'center': center.tolist(),
        'angle_per_step': float(median_angle),
        'axis_dispersion': float(axis_dispersion),
        'n_good_pairs': len(good_idx),
        'n_total_pairs': len(angles),
        'pair_details': [(int(i), float(a), float(r), float(f), ax.tolist())
                         for i, a, r, f, ax in pair_results],
    }

    with open(OUTPUT_DIR / "axis_estimate_v3.json", 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(
        OUTPUT_DIR / "axis_estimate_v3.npz",
        axis=median_axis,
        center=center,
        angle_per_step=median_angle,
        angles=np.array(angles),
        axes=np.array(axes_good_aligned),
        n_good=len(good_idx),
    )

    print(f"\n  Saved to {OUTPUT_DIR / 'axis_estimate_v3.*'}")

    # Quality assessment
    print("\n[Quality Assessment]")
    is_mainly_y = abs(median_axis[1]) > 0.5
    angle_ok = abs(median_angle - expected_angle) < 5
    dispersion_ok = axis_dispersion < 15
    enough_pairs = len(good_idx) >= 15

    print(f"  Mainly Y-axis: {is_mainly_y} ({median_axis[1]:.3f})")
    print(f"  Angle correct: {angle_ok} ({median_angle:.1f} vs {expected_angle:.1f})")
    print(f"  Axis consistent: {dispersion_ok} ({axis_dispersion:.1f} deg)")
    print(f"  Enough pairs: {enough_pairs} ({len(good_idx)}/24)")

    if is_mainly_y and angle_ok and dispersion_ok and enough_pairs:
        print("\n  *** HIGH CONFIDENCE: Axis estimate is reliable ***")
    elif is_mainly_y and angle_ok:
        print("\n  ** MEDIUM CONFIDENCE: Axis direction OK but needs validation **")
    else:
        print("\n  * LOW CONFIDENCE: Axis estimate may be unreliable *")

    return results


if __name__ == "__main__":
    main()
