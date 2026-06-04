"""
Utilities: depth→point cloud, pot segmentation, visualization.
"""
import numpy as np
import cv2
from PIL import Image
import tifffile

# Kinect V2 depth intrinsics (512×424)
FX_D, FY_D = 365.0, 365.0
CX_D, CY_D = 255.5, 211.5
DEPTH_H, DEPTH_W = 424, 512

# Kinect V2 color intrinsics (1920×1080)
FX_C, FY_C = 1080.0, 1080.0
CX_C, CY_C = 959.5, 539.5
COLOR_H, COLOR_W = 1080, 1920


def load_depth(depth_path):
    """Load depth image (mm uint16) → float meters (424, 512)."""
    img = tifffile.imread(depth_path)
    return img.astype(np.float64) / 1000.0  # mm → m


def load_color(color_path):
    """Load color image (1920, 1080, 3) uint8 BGR."""
    img = tifffile.imread(color_path)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def depth_to_xyz(depth_m, fx=FX_D, fy=FY_D, cx=CX_D, cy=CY_D):
    """
    Convert depth map to 3D points in depth camera frame.
    Returns (H*W, 3) float64 array.
    """
    h, w = depth_m.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    x = (uu - cx) * depth_m / fx
    y = (vv - cy) * depth_m / fy
    z = depth_m

    valid = (z > 0.2) & (z < 3.0) & np.isfinite(z)

    return np.column_stack([x[valid], y[valid], z[valid]]), valid, uu, vv


def colorize_points(xyz_cam, depth_m, color_img, valid_mask, uu, vv):
    """
    Project depth points to color camera and retrieve colors.
    R_d2c ≈ eye(3), t_d2c ≈ [-0.052, 0, 0] (Kinect V2 default).
    """
    R_d2c = np.eye(3)
    t_d2c = np.array([-0.052, 0.0, 0.0])

    pts_c = (R_d2c @ xyz_cam.T + t_d2c.reshape(3, 1)).T
    u_c = pts_c[:, 0] / pts_c[:, 2] * FX_C + CX_C
    v_c = pts_c[:, 1] / pts_c[:, 2] * FY_C + CY_C

    u_px = np.round(u_c).astype(int)
    v_px = np.round(v_c).astype(int)

    valid_proj = (pts_c[:, 2] > 0) & np.isfinite(u_c) & np.isfinite(v_c) & \
                 (u_px >= 0) & (u_px < COLOR_W) & (v_px >= 0) & (v_px < COLOR_H)

    colors = np.zeros((len(xyz_cam), 3), dtype=np.float64)
    idx_valid = np.where(valid_proj)[0]
    colors[idx_valid] = color_img[v_px[idx_valid], u_px[idx_valid]].astype(np.float64) / 255.0
    colors[idx_valid] = colors[idx_valid, ::-1]  # RGB → BGR correction? Actually just keep as-is

    return colors, valid_proj


def build_point_cloud(depth_path, color_path, depth_min=0.32, depth_max=2.0):
    """
    Full pipeline: depth → xyz → color → filtered point cloud.
    Returns dict with xyz (N,3), rgb (N,3), or None.
    """
    depth_m = load_depth(depth_path)
    color_img = load_color(color_path)

    # Median filter depth
    depth_filtered = cv2.medianBlur(depth_m.astype(np.float32), 3)

    xyz_cam, valid_mask, uu, vv = depth_to_xyz(depth_filtered)
    if len(xyz_cam) < 1000:
        return None

    rgb, valid_proj = colorize_points(xyz_cam, depth_filtered, color_img, valid_mask, uu, vv)

    # Combine masks
    keep = valid_proj & (xyz_cam[:, 2] > depth_min) & (xyz_cam[:, 2] < depth_max)

    return {
        'xyz': xyz_cam[keep],
        'rgb': rgb[keep],
        'depth_m': depth_filtered,
        'depth_valid': valid_mask,
        'n_total': len(xyz_cam),
        'n_kept': keep.sum(),
    }


def segment_pot(pcd_dict, y_thresh_offset=0.0):
    """
    Segment pot region: bottom part + non-green colors.

    Uses ExG (Excess Green) to identify leaves, then takes the complement
    for the pot. Additionally filters by Y height.

    Returns mask for pot points.
    """
    xyz = pcd_dict['xyz']
    rgb = pcd_dict['rgb']

    R, G, B = rgb[:, 2], rgb[:, 1], rgb[:, 0]  # BGR order

    # ExG > 0 means green (leaf)
    exg = 2 * G - R - B

    # Simple leaf mask
    leaf_mask = exg > 0.02

    # Y-based: pot is below median Y
    y_median = np.median(xyz[:, 1])
    y_low = xyz[:, 1] < (y_median + y_thresh_offset)

    # Pot = not leaf AND low Y
    pot_mask = (~leaf_mask) & y_low

    return pot_mask


def extract_roi(xyz, center, radius=0.15):
    """Extract points within radius of center in XZ plane."""
    dx = xyz[:, 0] - center[0]
    dz = xyz[:, 2] - center[2]
    dist = np.sqrt(dx**2 + dz**2)
    return dist < radius


def stats_str(arr, name=""):
    """Pretty-print statistics."""
    return f"{name}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, min={np.min(arr):.4f}, max={np.max(arr):.4f}"


# --- Open3D helpers ---
import open3d as o3d


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
    """Downsample + statistical outlier removal on Open3D point cloud."""
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
