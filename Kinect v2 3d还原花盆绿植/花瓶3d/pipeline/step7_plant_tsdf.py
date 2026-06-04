"""
Step 7: Full-scene TSDF reconstruction (voxel=3mm, trunc=15mm).

Integrates all 36 RGBD frames using pure rotation poses.
The TSDF naturally fuses all views into a combined mesh.
Plant regions are reconstructed wherever depth is consistent across frames.
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
    rot_around_center, visualize_mesh, save_mesh
)
from step1_align import (
    build_undistort_maps, build_alignment_lut,
    undistort_depth, compute_alignment_with_depth
)


def main():
    print("=" * 60)
    print("STEP 7: Full-scene TSDF reconstruction")
    print("=" * 60)

    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=TSDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)

    for i in range(NUM_FRAMES):
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(
            depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)

        depth_img = o3d.geometry.Image(depth_filtered_mm.astype(np.uint16))
        color_bgr = cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR)
        color_img = o3d.geometry.Image(color_bgr.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1000.0, depth_trunc=TSDF_DEPTH_MAX,
            convert_rgb_to_intensity=False)

        # Pure rotation pose (from Step 5/6 — consistent with plant stack)
        world_pose = rot_around_center(i * DEG_PER_FRAME)
        cam_pose = np.linalg.inv(world_pose)

        tsdf.integrate(rgbd, intrinsic, cam_pose)

        if i % 6 == 0:
            print(f"  Frame {i:03d} integrated")

    print("\nExtracting mesh...")
    mesh = tsdf.extract_triangle_mesh()

    mesh = mesh.simplify_vertex_clustering(
        VOXEL_SIZE, contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()

    verts = np.asarray(mesh.vertices)
    fin = np.all(np.isfinite(verts), axis=1)
    if not fin.all():
        mesh.remove_vertices_by_mask(~fin)

    # No Taubin — leave for step 8 fusion
    mesh.compute_vertex_normals()

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    out_path = os.path.join(OUTPUT_DIR, "plant_mesh.ply")
    save_mesh(out_path, mesh)
    print(f"Saved: {out_path}")

    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
