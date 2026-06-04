"""
Shared utility functions for the 3D reconstruction pipeline.
"""
import numpy as np
import cv2
import open3d as o3d
import os
import json

from config import *


# ============================================================
# Image I/O
# ============================================================

def load_depth(frame_idx):
    """Load depth image from file. Returns uint16 mm values (512x424)."""
    raw = np.fromfile(
        os.path.join(DEPTH_DIR, f"{frame_idx:03d}.png"), dtype=np.uint8)
    depth = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to load depth/{frame_idx:03d}.png")
    return depth


def load_color_bgr(frame_idx):
    """Load color image as BGR (1920x1080)."""
    raw = np.fromfile(
        os.path.join(COLOR_DIR, f"{frame_idx:03d}.png"), dtype=np.uint8)
    color = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Failed to load color/{frame_idx:03d}.png")
    return color


def load_color_rgb(frame_idx):
    """Load color image as RGB (1920x1080)."""
    bgr = load_color_bgr(frame_idx)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_ply(filepath, points, colors=None, normals=None):
    """Save a point cloud to PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(filepath, pcd)


def save_mesh(filepath, mesh):
    """Save a triangle mesh."""
    o3d.io.write_triangle_mesh(filepath, mesh)


# ============================================================
# Depth filtering
# ============================================================

def bilateral_filter_depth(depth_mm):
    """Apply bilateral filter to depth image. Input/output in mm."""
    depth_float = depth_mm.astype(np.float32)
    filtered = cv2.bilateralFilter(
        depth_float, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    return filtered


# ============================================================
# Point cloud generation
# ============================================================

def depth_to_points(depth_m, colors_uint8=None):
    """
    Unproject depth image (meters) to 3D points using depth camera intrinsics.

    Args:
        depth_m: (H, W) float array, depth in meters
        colors_uint8: (H, W, 3) uint8 RGB array, same resolution as depth

    Returns:
        pts: (N, 3) float array of 3D points
        clr: (N, 3) float array of colors [0,1], or None
    """
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h, 0:w]

    valid = (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M)
    z = depth_m[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)

    x = (u - DEPTH_CX) * z / DEPTH_FX
    y = (v - DEPTH_CY) * z / DEPTH_FY

    pts = np.stack([x, y, z], axis=1).astype(np.float64)

    clr = None
    if colors_uint8 is not None:
        c = colors_uint8[valid].astype(np.float64) / 255.0
        clr = c

    return pts, clr


# ============================================================
# Rotation around center (Y-axis)
# ============================================================

def rot_around_center(angle_deg):
    """
    4x4 transformation matrix: rotate around Y-axis through (ROT_CX, 0, ROT_CZ).
    Positive angle = counter-clockwise when viewed from above.
    """
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    T1 = np.eye(4)
    T1[0, 3] = -ROT_CX
    T1[2, 3] = -ROT_CZ

    R = np.eye(4)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c

    T2 = np.eye(4)
    T2[0, 3] = ROT_CX
    T2[2, 3] = ROT_CZ

    return T2 @ R @ T1


def rotate_points_around_center(pts, angle_deg):
    """
    Rotate Nx3 points around Y-axis through (ROT_CX, 0, ROT_CZ).
    Returns Nx3 points.
    """
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    pts_c = pts.copy()
    pts_c[:, 0] -= ROT_CX
    pts_c[:, 2] -= ROT_CZ

    x = pts_c[:, 0] * c + pts_c[:, 2] * s
    z = -pts_c[:, 0] * s + pts_c[:, 2] * c

    pts_c[:, 0] = x + ROT_CX
    pts_c[:, 2] = z + ROT_CZ

    return pts_c


# ============================================================
# Segmentation
# ============================================================

def is_green_hsv(hsv_img):
    """
    Determine if pixels are green based on HSV values.

    Args:
        hsv_img: (N, 3) uint8 HSV image (OpenCV format, H∈[0,180])

    Returns:
        (N,) bool mask
    """
    h = hsv_img[:, 0].astype(float)
    s = hsv_img[:, 1].astype(float)
    return (h >= GREEN_H_MIN) & (h <= GREEN_H_MAX) & (s > GREEN_S_MIN)


# ============================================================
# Filtering helpers
# ============================================================

def sor_filter(pcd, nb_neighbors, std_ratio):
    """Statistical Outlier Removal. Returns filtered point cloud."""
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def radius_filter(pcd, nb_points, radius):
    """Radius outlier removal. Returns filtered point cloud."""
    cl, _ = pcd.remove_radius_outlier(
        nb_points=nb_points, radius=radius)
    return cl


def voxel_downsample(pcd, voxel_size=None):
    """Voxel grid downsample."""
    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    return pcd.voxel_down_sample(voxel_size)


def estimate_normals(pcd, radius=None, max_nn=None):
    """Estimate normals for a point cloud."""
    if radius is None:
        radius = NORMAL_RADIUS
    if max_nn is None:
        max_nn = NORMAL_MAX_NN
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd


# ============================================================
# ICP helpers
# ============================================================

def icp_point_to_plane(source, target, init_transform, threshold, max_iter=50):
    """
    Point-to-plane ICP between two point clouds.
    Both source and target MUST have normals estimated.
    """
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter))
    return result


def icp_point_to_point(source, target, init_transform, threshold, max_iter=50):
    """
    Point-to-point ICP between two point clouds.
    Normals NOT required.
    """
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter))
    return result


# ============================================================
# Visualization helpers
# ============================================================

def visualize_point_clouds(geometries, window_name="Point Cloud"):
    """Show point clouds in Open3D visualizer. Blocks until window is closed."""
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        [axis] + geometries,
        window_name=window_name,
        width=1280, height=720)


def visualize_mesh(mesh, window_name="Mesh"):
    """Show mesh with back-face rendering."""
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        width=1280, height=720,
        mesh_show_back_face=True)


def make_colored_pcd(pts, color):
    """Create a point cloud with uniform color."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    clr = np.tile(np.array(color), (len(pts), 1))
    pcd.colors = o3d.utility.Vector3dVector(clr)
    return pcd


# ============================================================
# Metadata output
# ============================================================

def save_pose_json(poses, filepath):
    """
    Save per-frame poses to JSON.

    Args:
        poses: list of 4x4 numpy arrays (world-from-camera transforms)
        filepath: output JSON path
    """
    data = {
        "rotation_center_x": ROT_CX,
        "rotation_center_z": ROT_CZ,
        "deg_per_frame": DEG_PER_FRAME,
        "total_frames": NUM_FRAMES,
        "frames": []
    }
    for i, pose in enumerate(poses):
        raw_angle = i * DEG_PER_FRAME
        # Extract rotation angle from the actual pose
        R = pose[:3, :3]
        actual_angle = np.degrees(np.arctan2(R[0, 2], R[0, 0]))
        t = pose[:3, 3]
        data["frames"].append({
            "frame": i,
            "angle_deg": round(raw_angle, 1),
            "actual_angle_deg": round(float(actual_angle), 3),
            "rotation_matrix": R.tolist(),
            "translation": t.tolist()
        })
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_report(report, filepath):
    """Save reconstruction report to JSON."""
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
