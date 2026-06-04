"""
Step 1: Depth-to-color alignment (replaces pylibfreenect2 Registration).

Uses OpenCV + known Kinect V2 calibration to:
1. Undistort depth image (radial distortion)
2. Back-project depth pixels to 3D (depth camera frame)
3. Transform to color camera frame (extrinsic)
4. Project to color image plane → sample color
5. Output: 512x424 aligned depth + 512x424 aligned color

Equivalent to pylibfreenect2 Registration.apply() output:
  - undistorted: 512x424 float depth (meters)
  - registered:  512x424 uint8 RGB
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import load_depth, load_color_rgb, visualize_point_clouds, depth_to_points, save_ply


def build_undistort_maps():
    """
    Build the rectification maps for the depth camera.
    Returns map1, map2 for cv2.remap.
    """
    # We want to undistort: the output has the same camera matrix
    map1, map2 = cv2.initUndistortRectifyMap(
        K_DEPTH, D_DEPTH, None, K_DEPTH,
        (DEPTH_WIDTH, DEPTH_HEIGHT), cv2.CV_32FC1)
    return map1, map2


def undistort_depth(depth_mm, map1, map2):
    """
    Undistort depth image (mm → mm).
    Uses precomputed rectification maps for speed.
    """
    depth_float = depth_mm.astype(np.float32)
    undistorted = cv2.remap(depth_float, map1, map2, cv2.INTER_NEAREST)
    return undistorted


def build_alignment_lut():
    """
    Build lookup tables mapping (u_d, v_d) depth pixel → (u_c, v_c) color pixel.

    Steps:
    1. Back-project depth pixel to 3D (depth camera coords)
    2. Transform to color camera coords
    3. Project to color image

    Returns:
        lut_u: (424, 512) int32, color image u-coordinate (-1 = invalid)
        lut_v: (424, 512) int32, color image v-coordinate (-1 = invalid)
    """
    h, w = DEPTH_HEIGHT, DEPTH_WIDTH  # 424, 512

    # Create pixel grids
    vv, uu = np.mgrid[0:h, 0:w]

    # Back-project to 3D with unit depth (z=1)
    x_unit = (uu - DEPTH_CX) / DEPTH_FX
    y_unit = (vv - DEPTH_CY) / DEPTH_FY

    # These are 3D points at z=1 (normalized coords)
    pts_unit = np.stack([x_unit, y_unit, np.ones_like(x_unit)], axis=-1)  # (424, 512, 3)

    # Transform to color camera frame: P_color = R @ P_depth + T
    # Since R ≈ I, T ≈ [-0.052, 0, 0], this shifts x by -0.052m at z=1
    pts_color = pts_unit.copy()
    pts_color[..., 0] += T_DEPTH_TO_COLOR[0]  # x shift (at unit depth)
    pts_color[..., 1] += T_DEPTH_TO_COLOR[1]
    pts_color[..., 2] += T_DEPTH_TO_COLOR[2]

    # Project to color image plane: u = fx * x/z + cx, v = fy * y/z + cy
    z_c = pts_color[..., 2]
    u_c = (pts_color[..., 0] * COLOR_FX / z_c + COLOR_CX).round().astype(np.int32)
    v_c = (pts_color[..., 1] * COLOR_FY / z_c + COLOR_CY).round().astype(np.int32)

    # Mask invalid projections
    valid = (
        (u_c >= 0) & (u_c < COLOR_WIDTH) &
        (v_c >= 0) & (v_c < COLOR_HEIGHT)
    )

    lut_u = np.where(valid, u_c, -1)
    lut_v = np.where(valid, v_c, -1)

    return lut_u, lut_v


def compute_alignment_with_depth(depth_m, lut_u, lut_v, color_rgb):
    """
    Compute actual aligned color for each depth pixel given real depth values.

    The baseline shift at z=1 (in build_alignment_lut) needs correction for actual depth.
    For a point at depth z, its x-coordinate in color camera frame is:
      x_color = (x_depth_unit * z) + T_x
    Projected u-coordinate:
      u_c = (x_color / z) * COLOR_FX + COLOR_CX
          = (x_depth_unit * z + T_x) / z * COLOR_FX + COLOR_CX
          = (x_depth_unit + T_x/z) * COLOR_FX + COLOR_CX

    So the LUT built at z=1 uses T_x/1, but at real depth z it should use T_x/z.
    The correction: u_c(z) = u_c(z=1) + COLOR_FX * T_x * (1/z - 1)

    Args:
        depth_m: (424, 512) float, depth in meters
        lut_u: (424, 512) int32, precomputed color u at z=1
        lut_v: (424, 512) int32, precomputed color v at z=1
        color_rgb: (1080, 1920, 3) uint8 RGB

    Returns:
        aligned_color: (424, 512, 3) uint8 RGB
    """
    h, w = DEPTH_HEIGHT, DEPTH_WIDTH

    # Start with the z=1 LUT
    u_c = lut_u.astype(np.float64)
    v_c = lut_v.astype(np.float64)

    # Apply depth-dependent correction for the baseline
    # u_corrected = u_LUT + fx_color * T_x * (1/z - 1)
    valid_depth = (depth_m > 0.01)
    inv_z = np.zeros_like(depth_m)
    inv_z[valid_depth] = 1.0 / depth_m[valid_depth]

    u_c[valid_depth] += COLOR_FX * T_DEPTH_TO_COLOR[0] * (inv_z[valid_depth] - 1.0)
    v_c[valid_depth] += COLOR_FY * T_DEPTH_TO_COLOR[1] * (inv_z[valid_depth] - 1.0)

    # Round and clip
    u_c = np.round(u_c).astype(np.int32)
    v_c = np.round(v_c).astype(np.int32)
    valid = (
        (u_c >= 0) & (u_c < COLOR_WIDTH) &
        (v_c >= 0) & (v_c < COLOR_HEIGHT) &
        (lut_u >= 0)  # also must be valid in original LUT
    )

    # Build aligned color image
    aligned_color = np.zeros((h, w, 3), dtype=np.uint8)
    aligned_color[valid] = color_rgb[v_c[valid], u_c[valid]]

    return aligned_color


# ============================================================
# Main
# ============================================================
def process_frame(frame_idx, map1, map2, lut_u, lut_v):
    """Process one frame: load → undistort → align."""
    depth_mm = load_depth(frame_idx)
    color_rgb = load_color_rgb(frame_idx)

    # Undistort depth
    depth_undist_mm = undistort_depth(depth_mm, map1, map2)
    depth_undist_m = depth_undist_mm / 1000.0

    # Align color to depth
    aligned_color = compute_alignment_with_depth(
        depth_undist_m, lut_u, lut_v, color_rgb)

    return depth_undist_m, aligned_color


def main():
    print("=" * 60)
    print("STEP 1: Depth-to-color alignment")
    print("=" * 60)

    # Precompute maps and LUTs (done once)
    print("Building undistortion maps...")
    map1, map2 = build_undistort_maps()

    print("Building alignment LUT...")
    lut_u, lut_v = build_alignment_lut()
    valid_lut_pct = np.sum(lut_u >= 0) / (DEPTH_WIDTH * DEPTH_HEIGHT) * 100
    print(f"  LUT coverage: {valid_lut_pct:.1f}% of depth pixels map to color FOV")

    # Process frame 0 for verification
    print("\nProcessing frame 0 for verification...")
    depth_m, aligned_color = process_frame(0, map1, map2, lut_u, lut_v)

    # Save aligned frame 0
    out_dir = os.path.join(OUTPUT_DIR, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth_000.npy"), depth_m.astype(np.float32))
    cv2.imwrite(os.path.join(out_dir, "color_000.png"),
                cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR))

    # Generate point cloud for visual verification
    pts, clr = depth_to_points(depth_m, aligned_color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(clr)

    save_ply(os.path.join(out_dir, "frame_000_aligned.ply"), pts, clr)

    print(f"  3D points: {len(pts)}")
    print(f"  Depth range: [{depth_m[depth_m>0].min():.3f}, {depth_m[depth_m>0].max():.3f}] m")
    print(f"  Color range: [{aligned_color.min()}, {aligned_color.max()}]")

    # Visualize
    print("\nVerification: aligned point cloud for frame 0")
    print("  Pot edges should align with color edges (no color fringing)")
    print("  Close window to continue...")
    visualize_point_clouds(
        [pcd],
        window_name="Step 1 — Aligned Frame 0 (close to confirm)")

    print("\nStep 1 complete. Output: pipeline_output/aligned/")
    print("  depth_000.npy  — 512x424 undistorted depth (meters)")
    print("  color_000.png  — 512x424 registered color")
    print("  frame_000_aligned.ply — 3D point cloud for verification")


if __name__ == "__main__":
    main()
