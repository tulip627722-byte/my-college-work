"""
Step 4: Pot registration (PointToPlane ICP + Pose Graph) + TSDF reconstruction.

For each frame's pot point cloud:
1. Initial pose: rot_around_center(i * 10°), center=(-0.0346, 0.7861)
2. Adjacent-frame PointToPlane ICP (threshold=0.05m)
3. Pose graph optimization: 35 adjacent edges + 1 loop closure edge
4. TSDF fusion with optimized poses
5. Extract mesh, vertex clustering, Taubin smooth x3

Output: pipeline_output/pot_mesh.ply
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, bilateral_filter_depth,
    rot_around_center, sor_filter, voxel_downsample, estimate_normals,
    visualize_mesh, save_mesh
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth
from step3_segment import segment_frame


def extract_pot_clouds():
    """Extract pot point clouds from all 36 frames."""
    print("Extracting pot clouds from all frames...")
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    pot_pcds = []

    for i in range(NUM_FRAMES):
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        pot_pcd, _ = segment_frame(depth_filtered_m, aligned_color)
        pot_pcds.append(pot_pcd)

        if i % 6 == 0:
            print(f"  Frame {i:03d}: pot={len(pot_pcd.points)} pts")

    return pot_pcds


def register_pot_p2plane(pot_pcds):
    """
    Register pot clouds using adjacent-frame PointToPlane ICP.

    Returns:
        world_poses: list of 4x4 numpy arrays (world-from-camera for each frame)
    """
    print("\n" + "-" * 40)
    print("Adjacent-frame PointToPlane ICP...")
    print("-" * 40)

    world_poses = [np.eye(4)]  # Frame 0 at identity

    for i in range(1, NUM_FRAMES):
        src = pot_pcds[i]
        tgt = pot_pcds[i - 1]

        if len(src.points) < 30 or len(tgt.points) < 30:
            print(f"  Frame {i:03d}: too few points, using nominal pose")
            T = rot_around_center(-DEG_PER_FRAME)
        else:
            # Initial guess: undo turntable rotation
            init = rot_around_center(-DEG_PER_FRAME)

            try:
                reg = o3d.pipelines.registration.registration_icp(
                    src, tgt, POT_ICP_THRESHOLD, init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6,
                        max_iteration=POT_ICP_MAX_ITER))
                T = reg.transformation
                print(f"  Frame {i:03d}: fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}m "
                      f"| src={len(src.points)} tgt={len(tgt.points)}")
            except Exception as e:
                print(f"  Frame {i:03d}: ICP failed ({e}), using nominal")
                T = init

        world_poses.append(world_poses[-1] @ T)

    return world_poses


def optimize_pose_graph(pot_pcds, world_poses):
    """
    Build and optimize pose graph with loop closure.
    """
    print("\n" + "-" * 40)
    print("Pose graph optimization...")
    print("-" * 40)

    pg = o3d.pipelines.registration.PoseGraph()

    # Add nodes
    for i in range(NUM_FRAMES):
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(world_poses[i]))

    # Add adjacent edges
    for i in range(1, NUM_FRAMES):
        T_edge = np.linalg.inv(world_poses[i - 1]) @ world_poses[i]
        src = pot_pcds[i]
        tgt = pot_pcds[i - 1]

        if len(src.points) >= 30 and len(tgt.points) >= 30:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                tgt, src, POT_ICP_THRESHOLD, T_edge)
        else:
            info = np.eye(6) * 0.1

        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i - 1, i, T_edge, info, uncertain=False))

    # Loop closure: frame 35 → frame 0
    print("  Computing loop closure (frame 35 → frame 0)...")
    src_loop = pot_pcds[NUM_FRAMES - 1]
    tgt_loop = pot_pcds[0]
    init_loop = rot_around_center(DEG_PER_FRAME)  # rotate forward by 10°

    try:
        reg_lc = o3d.pipelines.registration.registration_icp(
            src_loop, tgt_loop, POT_ICP_THRESHOLD, init_loop,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=POT_ICP_MAX_ITER))
        info_lc = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            tgt_loop, src_loop, POT_ICP_THRESHOLD, reg_lc.transformation)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            NUM_FRAMES - 1, 0, reg_lc.transformation, info_lc, uncertain=False))
        print(f"  Loop closure: fit={reg_lc.fitness:.3f} rmse={reg_lc.inlier_rmse:.4f}m")
    except Exception as e:
        print(f"  Loop closure skipped: {e}")

    # Optimize
    print("  Running pose graph optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=POT_ICP_THRESHOLD,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    opt_poses = [node.pose for node in pg.nodes]

    # Print summary
    angle_18 = np.degrees(np.arctan2(opt_poses[18][0, 2], opt_poses[18][0, 0]))
    angle_35 = np.degrees(np.arctan2(opt_poses[35][0, 2], opt_poses[35][0, 0]))
    print(f"  Optimized: frame 18 angle={angle_18:.1f}° (expect ~180°), "
          f"frame 35 angle={angle_35:.1f}° (expect ~350°)")

    return opt_poses


def tsdf_pot(opt_poses):
    """
    Build TSDF volume from pot point clouds using optimized poses.

    Integrates each frame's point cloud into TSDF by:
    1. Transforming points to world frame
    2. Computing camera pose from world pose
    3. Integrating the RGBD image
    """
    print("\n" + "-" * 40)
    print("TSDF pot reconstruction (voxel=3mm, trunc=15mm)...")
    print("-" * 40)

    # Build per-frame data
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=TSDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)

    for i in range(NUM_FRAMES):
        # Load and prepare data
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)

        # Create RGBD image
        depth_img = o3d.geometry.Image(depth_filtered_mm.astype(np.uint16))
        color_bgr = cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR)
        color_img = o3d.geometry.Image(color_bgr.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1000.0,
            depth_trunc=TSDF_DEPTH_MAX,
            convert_rgb_to_intensity=False)

        # Camera pose = inverse of world-from-camera
        world_pose = opt_poses[i]
        cam_pose = np.linalg.inv(world_pose)

        tsdf.integrate(rgbd, intrinsic, cam_pose)

        if i % 6 == 0:
            print(f"  Frame {i:03d} integrated")

    # Extract mesh
    print("\nExtracting pot mesh...")
    pot_mesh = tsdf.extract_triangle_mesh()

    # Cleanup
    pot_mesh = pot_mesh.simplify_vertex_clustering(
        VOXEL_SIZE, contraction=o3d.geometry.SimplificationContraction.Average)
    pot_mesh = pot_mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERATIONS)
    pot_mesh.remove_degenerate_triangles()
    pot_mesh.remove_duplicated_vertices()

    print(f"Pot mesh: {len(pot_mesh.vertices)} vertices, {len(pot_mesh.triangles)} triangles")

    return pot_mesh


def main():
    print("=" * 60)
    print("STEP 4: Pot registration + TSDF reconstruction")
    print("=" * 60)

    # Extract pot point clouds
    pot_pcds = extract_pot_clouds()

    # Register with PointToPlane ICP
    world_poses = register_pot_p2plane(pot_pcds)

    # Pose graph optimization
    opt_poses = optimize_pose_graph(pot_pcds, world_poses)

    # TSDF reconstruction
    pot_mesh = tsdf_pot(opt_poses)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "pot_mesh.ply")
    save_mesh(out_path, pot_mesh)
    print(f"\nSaved: {out_path}")

    # Verify
    print("\nVerification: pot mesh — close window to confirm")
    print("  Pot should be complete, no ghosting, no distortion")
    visualize_mesh(pot_mesh, "Step 4 — Pot Mesh (close to confirm)")

    print("\nStep 4 complete. pot_mesh.ply is LOCKED — do not re-run.")


if __name__ == "__main__":
    main()
