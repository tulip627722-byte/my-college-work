"""
Step 2: Per-frame 3D point cloud generation.

For each frame:
1. Load depth + color
2. Undistort depth (radial correction)
3. Align color to depth (extrinsic transform)
4. Bilateral filter depth
5. Filter depth range 0.3 ~ 1.5m
6. Unproject to 3D point cloud (depth camera intrinsics)

Output: 36 colored point clouds in pipeline_output/frames/
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, depth_to_points, save_ply,
    bilateral_filter_depth, visualize_point_clouds, make_colored_pcd
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth


def process_all_frames():
    """Process all 36 frames: align → filter → unproject → save."""
    out_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(out_dir, exist_ok=True)

    # Precompute maps (done once for all frames)
    print("Building undistortion maps and alignment LUT...")
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    point_clouds = []

    for i in range(NUM_FRAMES):
        # Load raw
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)

        # Undistort depth
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)

        # Align color to depth
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(
            depth_undist_m, lut_u, lut_v, color_rgb)

        # Bilateral filter depth
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        # Generate 3D point cloud
        pts, clr = depth_to_points(depth_filtered_m, aligned_color)

        # Save
        save_ply(
            os.path.join(out_dir, f"frame_{i:03d}.ply"), pts, clr)

        point_clouds.append((pts, clr))

        if i % 6 == 0:
            print(f"  Frame {i:03d}: {len(pts)} points "
                  f"(depth range [{depth_filtered_m[depth_filtered_m>0].min():.3f}, "
                  f"{depth_filtered_m[depth_filtered_m>0].max():.3f}] m)")

    print(f"\nSaved {NUM_FRAMES} point clouds to {out_dir}")
    return point_clouds


def main():
    print("=" * 60)
    print("STEP 2: Per-frame 3D point cloud generation")
    print("=" * 60)

    point_clouds = process_all_frames()

    # Verify: visualize 3 random frames
    print("\nVerification: visualizing frames 0, 12, 24...")
    print("  Pot should be white, plant should be green")

    for fidx in [0, 12, 24]:
        pts, clr = point_clouds[fidx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(clr)
        print(f"\n  Frame {fidx}: {len(pts)} points — close window to continue")
        visualize_point_clouds(
            [pcd],
            window_name=f"Step 2 — Frame {fidx:03d} (close to continue)")

    print("\nStep 2 complete.")
    print(f"  {NUM_FRAMES} point clouds → pipeline_output/frames/")


if __name__ == "__main__":
    main()
