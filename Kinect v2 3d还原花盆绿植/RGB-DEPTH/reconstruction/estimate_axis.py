"""
Core: Estimate rotation axis from 25-frame turntable data.

Strategy:
1. Load all frames, segment pot (rigid region)
2. FGR global registration on adjacent pot clouds (no axis prior needed)
3. From transforms {R_i, t_i}, solve for rotation axis direction a and center c
4. Validate: angle per step should ≈14.4°
5. Cross-check transforms predicted by axis vs FGR results
"""
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as Rsp
import pickle

from utils import (
    build_point_cloud, segment_pot, extract_roi, stats_str
)

DATA_DIR = Path("E:/RGB-DEPTH/kinect_sync_data_manual")
OUTPUT_DIR = Path("E:/RGB-DEPTH/reconstruction")
N_FRAMES = 25


def numpy_to_o3d(xyz, rgb=None):
    """Convert numpy arrays to Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pcd


def o3d_to_numpy(pcd):
    """Extract xyz, rgb from Open3D point cloud."""
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    return xyz, rgb


def preprocess_pcd(pcd, voxel_size=0.003, nb_neighbors=20, std_ratio=1.5):
    """Downsample + statistical outlier removal."""
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) < 50:
        return pcd
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return pcd


def compute_normals(pcd, radius=0.01):
    """Estimate normals for FPFH computation."""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return pcd


def fgr_register(source, target, voxel_size=0.005):
    """
    Fast Global Registration on pot point clouds.
    No initial transform needed — rotation up to ~30° is fine.
    """
    # Downsample for feature computation
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return None, None

    # FPFH features
    radius_feature = voxel_size * 3
    src_down = compute_normals(src_down, radius_feature)
    tgt_down = compute_normals(tgt_down, radius_feature)

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature * 1.5, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature * 1.5, max_nn=100)
    )

    # FGR
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 4,
            iteration_number=128,
            use_absolute_scale=False,
            decrease_mu=True,
            maximum_tuple_count=1000,
        )
    )

    return result.transformation, result


def icp_refine(source, target, init_transform, voxel_size=0.002, max_dist=0.03):
    """Point-to-plane ICP refinement after coarse registration."""
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return init_transform, float('inf')

    src_down = compute_normals(src_down, voxel_size * 3)
    tgt_down = compute_normals(tgt_down, voxel_size * 3)

    try:
        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down, max_dist, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
            )
        )
        return result.transformation, result.inlier_rmse
    except Exception:
        return init_transform, float('inf')


def decompose_transform(T):
    """Extract R, t from 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def rotation_axis_angle_from_R(R):
    """
    Extract rotation axis direction and angle from rotation matrix.
    Returns (axis_unit_vector, angle_degrees).
    """
    # Axis from skew-symmetric part
    skew = R - R.T
    a = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    # Angle from trace
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)

    # Normalize axis
    a_norm = np.linalg.norm(a)
    if a_norm < 1e-10:
        return np.array([0, 0, 1]), 0.0

    a = a / a_norm
    return a, np.degrees(angle)


def solve_rotation_center(R_list, t_list):
    """
    Solve for rotation center c from: t_i = (I - R_i) * c.

    Stacks all equations into one linear system and solves with least squares.

    Args:
        R_list: list of (3,3) rotation matrices
        t_list: list of (3,) translation vectors

    Returns:
        c: estimated rotation center (3,)
        residuals: per-equation residuals
    """
    A = []  # Stack (I - R_i) → 3N × 3
    b = []  # Stack t_i → 3N

    for R, t in zip(R_list, t_list):
        A_i = np.eye(3) - R
        A.append(A_i)
        b.append(t)

    A = np.vstack(A)
    b = np.concatenate(b)

    # Least squares with regularization (Tikhonov for ill-conditioned cases)
    c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    # Per-equation residuals
    pred = A @ c
    per_eq_residuals = pred - b

    return c, per_eq_residuals


def align_axis_signs(axes):
    """Ensure all axis directions point in the same hemisphere."""
    ref = axes[0]
    aligned = [ref]
    for a in axes[1:]:
        if np.dot(a, ref) < 0:
            aligned.append(-a)
        else:
            aligned.append(a)
    return np.array(aligned)


def median_axis_angle(axes, angles):
    """
    Compute robust (median) axis direction and rotation angle.
    """
    # Align axis signs
    axes_aligned = align_axis_signs(axes)

    # Median axis (component-wise, then re-normalize)
    median_axis = np.median(axes_aligned, axis=0)
    median_axis = median_axis / np.linalg.norm(median_axis)

    # Median angle
    median_angle = np.median(angles)

    return median_axis, median_angle, axes_aligned


def build_pot_clouds():
    """
    Load all 25 frames, segment pot, return list of Open3D pot point clouds.
    """
    print("=" * 60)
    print("PHASE 1: Load 25 frames + segment pot region")
    print("=" * 60)

    all_pcds = []  # Full point clouds
    pot_pcds = []  # Pot-only point clouds

    for i in range(1, N_FRAMES + 1):
        depth_path = DATA_DIR / f"{i:04d}_depth.tif"
        color_path = DATA_DIR / f"{i:04d}_color.tif"

        if not depth_path.exists():
            print(f"  Frame {i:02d}: file not found, skipping")
            continue

        pcd_dict = build_point_cloud(str(depth_path), str(color_path))
        if pcd_dict is None or pcd_dict['n_kept'] < 1000:
            print(f"  Frame {i:02d}: insufficient points")
            continue

        # Segment pot
        pot_mask = segment_pot(pcd_dict, y_thresh_offset=0.02)

        xyz_full = pcd_dict['xyz']
        rgb_full = pcd_dict['rgb']

        pot_xyz = xyz_full[pot_mask]
        pot_rgb = rgb_full[pot_mask]

        if len(pot_xyz) < 500:
            print(f"  Frame {i:02d}: pot too small ({len(pot_xyz)} pts)")
            continue

        # Build Open3D clouds
        pcd_full = numpy_to_o3d(xyz_full, rgb_full)
        pcd_pot = numpy_to_o3d(pot_xyz, pot_rgb)

        # Quick preprocessing
        pcd_pot = preprocess_pcd(pcd_pot, voxel_size=0.002)

        all_pcds.append(pcd_full)
        pot_pcds.append(pcd_pot)

        print(f"  Frame {i:02d}: full={len(xyz_full)}, pot={len(pot_xyz)}, "
              f"pot_after_filter={len(pcd_pot.points)}")

    print(f"\nLoaded {len(pot_pcds)} frames with pot clouds")
    return all_pcds, pot_pcds


def register_adjacent_pairs(pot_pcds):
    """
    Phase 2: FGR + ICP on each adjacent pair of pot clouds.
    Returns list of transforms T_i mapping frame i to frame i-1.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Adjacent frame registration (FGR + ICP)")
    print("=" * 60)

    transforms = []  # T_i: frame i → frame i-1
    rotations = []   # R_i matrices
    translations = []  # t_i vectors
    angles = []      # rotation angles
    axes = []        # rotation axes
    rmses = []

    for i in range(1, len(pot_pcds)):
        source = pot_pcds[i]      # frame k (to be moved)
        target = pot_pcds[i - 1]  # frame k-1 (reference)

        print(f"\n  Pair {i+1:02d} → {i:02d}:")
        print(f"    Source (pot): {len(source.points)} pts, Target (pot): {len(target.points)} pts")

        # Step 1: FGR (no initial guess)
        T_fgr, fgr_result = fgr_register(source, target, voxel_size=0.004)

        if T_fgr is None or fgr_result is None:
            print(f"    FGR FAILED — trying ICP with identity")
            T_fgr = np.eye(4)

        fgr_R, fgr_t = decompose_transform(T_fgr)
        fgr_axis, fgr_angle = rotation_axis_angle_from_R(fgr_R)
        print(f"    FGR: angle={fgr_angle:.2f}°, axis=[{fgr_axis[0]:.3f}, {fgr_axis[1]:.3f}, {fgr_axis[2]:.3f}]")

        # Step 2: ICP refinement
        T_refined, rmse = icp_refine(source, target, T_fgr, voxel_size=0.002, max_dist=0.04)

        R, t = decompose_transform(T_refined)
        axis, angle = rotation_axis_angle_from_R(R)

        if rmse < 0.05:
            print(f"    ICP refined: angle={angle:.2f}°, RMSE={rmse:.5f}m ✓")
            transforms.append(T_refined)
            rotations.append(R)
            translations.append(t)
            angles.append(angle)
            axes.append(axis)
            rmses.append(rmse)
        else:
            # FGR-only fallback
            axis_f, angle_f = rotation_axis_angle_from_R(fgr_R)
            print(f"    ICP RMSE={rmse:.5f}m too high, using FGR-only: angle={angle_f:.2f}°")
            transforms.append(T_fgr)
            rotations.append(fgr_R)
            translations.append(fgr_t)
            angles.append(fgr_angle)
            axes.append(fgr_axis)
            rmses.append(float('inf'))

    return transforms, rotations, translations, angles, axes, rmses


def estimate_axis_from_transforms(rotations, translations, angles, axes):
    """
    Phase 3: Estimate rotation axis direction and center from pairwise transforms.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Rotation axis estimation")
    print("=" * 60)

    if len(rotations) < 5:
        print("ERROR: Too few successful registrations (< 5)")
        return None

    # 3a: Estimate axis direction (robust median)
    median_axis, median_angle, axes_aligned = median_axis_angle(axes, angles)

    # Angular consistency
    angle_std = np.std(angles)
    angle_expected = 360.0 / len(rotations) / (N_FRAMES / len(rotations))
    # Actually: intervals = number of pairs we got
    expected_angle = 360.0 / N_FRAMES

    print(f"\n  Axis direction (median): [{median_axis[0]:.4f}, {median_axis[1]:.4f}, {median_axis[2]:.4f}]")
    print(f"  Rotation angle (median): {median_angle:.2f}°")
    print(f"  Rotation angle (std):    {angle_std:.2f}°")
    print(f"  Expected angle/step:     {expected_angle:.1f}°")
    print(f"  Axis dispersion (deg):   {np.degrees(np.mean([np.arccos(np.clip(np.abs(np.dot(a, median_axis)), 0, 1)) for a in axes])):.2f}°")

    # 3b: Estimate rotation center from: t = (I - R) * c
    center, residuals = solve_rotation_center(rotations, translations)

    # Compute per-pair center estimates for statistics
    per_pair_centers = []
    for R, t in zip(rotations, translations):
        A = np.eye(3) - R
        try:
            ci = np.linalg.lstsq(A, t, rcond=None)[0]
            per_pair_centers.append(ci)
        except np.linalg.LinAlgError:
            pass

    per_pair_centers = np.array(per_pair_centers)

    if len(per_pair_centers) > 0:
        center_std = np.std(per_pair_centers, axis=0)
        print(f"\n  Rotation center:        [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")
        print(f"  Center std (per-pair):  [{center_std[0]:.4f}, {center_std[1]:.4f}, {center_std[2]:.4f}] m")
        print(f"  Residual norm (joint):  {np.linalg.norm(residuals):.4f}")

    # 3c: Quality check
    print(f"\n  Quality assessment:")
    if abs(median_angle - expected_angle) < 3.0:
        print(f"    ✓ Angle matches expected ({abs(median_angle - expected_angle):.1f}° off)")
    else:
        print(f"    ⚠ Angle deviates from expected by {abs(median_angle - expected_angle):.1f}°")

    axis_dispersion = np.degrees(np.mean([
        np.arccos(np.clip(np.abs(np.dot(a, median_axis)), 0, 1)) for a in axes
    ]))
    if axis_dispersion < 5.0:
        print(f"    ✓ Axis direction consistent ({axis_dispersion:.1f}° dispersion)")
    else:
        print(f"    ⚠ Axis direction scattered ({axis_dispersion:.1f}° dispersion)")

    return {
        'axis': median_axis,
        'angle_per_step': median_angle,
        'center': center,
        'center_std': center_std if len(per_pair_centers) > 0 else None,
        'residual_norm': np.linalg.norm(residuals),
        'per_pair_angles': angles,
        'per_pair_axes': axes_aligned,
        'per_pair_rmse': [],  # filled later
        'axis_dispersion': axis_dispersion,
        'n_pairs': len(rotations),
    }


def build_rotation_matrix(axis, angle_deg):
    """Build rotation matrix about arbitrary axis by given angle (degrees)."""
    return Rsp.from_rotvec(np.array(axis) * np.radians(angle_deg)).as_matrix()


def validate_axis(axis_info, pot_pcds):
    """
    Phase 4: Validate the estimated axis by predicting transforms
    and comparing with actual registration results.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Axis validation — theory vs actual transforms")
    print("=" * 60)

    axis = axis_info['axis']
    center = axis_info['center']
    angle_per_step = axis_info['angle_per_step']

    # Build theoretical transforms using estimated axis
    theoretical_errors = []
    for i in range(1, len(pot_pcds)):
        # Theoretical: rotation about axis through center by i * angle_per_step
        # But our transforms are frame k → frame k-1 (single step)
        R_theory = build_rotation_matrix(axis, angle_per_step)
        t_theory = (np.eye(3) - R_theory) @ center

        T_theory = np.eye(4)
        T_theory[:3, :3] = R_theory
        T_theory[:3, 3] = t_theory

        # Compare with estimate from axis_info if we have it
        if i - 1 < len(axis_info.get('per_pair_rotations', [])):
            R_actual = axis_info['per_pair_rotations'][i - 1]
            R_error = np.rad2deg(np.arccos(
                np.clip((np.trace(R_theory.T @ R_actual) - 1) / 2, -1, 1)
            ))
            theoretical_errors.append(R_error)

    if theoretical_errors:
        print(f"  Mean rotation error (theory vs actual): {np.mean(theoretical_errors):.2f}°")
        print(f"  Max  rotation error (theory vs actual): {np.max(theoretical_errors):.2f}°")

    return theoretical_errors


def main():
    print("ROTATION AXIS ESTIMATION — 25-frame turntable\n")

    # Phase 1: Load
    all_pcds, pot_pcds = build_pot_clouds()

    if len(pot_pcds) < 3:
        print("FATAL: Too few frames with valid pot clouds")
        return

    # Phase 2: Register adjacent pairs
    transforms, rotations, translations, angles, axes, rmses = \
        register_adjacent_pairs(pot_pcds)

    if len(rotations) < 5:
        print("\nFATAL: Too few successful registrations")
        print("Falling back to manual calibration or different approach.")
        print("\nDiagnostic: Check pot segmentation quality.")
        print("  - Is the pot clearly visible in depth images?")
        print("  - Does the pot have enough geometric texture for FPFH?")
        return

    # Phase 3: Estimate axis
    axis_info = estimate_axis_from_transforms(rotations, translations, angles, axes)
    axis_info['per_pair_rotations'] = rotations
    axis_info['per_pair_rmses'] = rmses

    # Phase 4: Validate
    validate_axis(axis_info, pot_pcds)

    # Save results
    output_file = OUTPUT_DIR / "axis_estimate.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(axis_info, f)
    print(f"\nAxis estimate saved to: {output_file}")

    # Also save for MATLAB compatibility
    np.savez(
        OUTPUT_DIR / "axis_estimate.npz",
        axis=axis_info['axis'],
        center=axis_info['center'],
        angle_per_step=axis_info['angle_per_step'],
        angles=np.array(axis_info['per_pair_angles']),
        axes=np.array(axis_info['per_pair_axes']),
        n_pairs=axis_info['n_pairs'],
    )
    print(f"NumPy arrays saved to: {OUTPUT_DIR / 'axis_estimate.npz'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Axis direction:  [{axis_info['axis'][0]:.4f}, {axis_info['axis'][1]:.4f}, {axis_info['axis'][2]:.4f}]")
    print(f"  Rotation center: [{axis_info['center'][0]:.4f}, {axis_info['center'][1]:.4f}, {axis_info['center'][2]:.4f}] m")
    print(f"  Angle per step:  {axis_info['angle_per_step']:.2f}°")
    print(f"  Successful pairs: {axis_info['n_pairs']}/{len(pot_pcds)-1}")

    return axis_info, pot_pcds, transforms


if __name__ == "__main__":
    main()
