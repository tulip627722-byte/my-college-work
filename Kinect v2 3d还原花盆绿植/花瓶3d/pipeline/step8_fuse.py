"""
Step 8: Pot + Plant fusion.

1. Load pot_mesh.ply and plant_mesh.ply
2. Merge meshes
3. Remove duplicate vertices
4. Remove degenerate triangles
5. Taubin smooth x3
6. Output: fused_mesh.ply + fused_mesh.obj

Does NOT re-run pot or plant reconstruction.
Pot mesh is treated as LOCKED.
"""
import numpy as np
import open3d as o3d
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import visualize_mesh, save_mesh


def main():
    print("=" * 60)
    print("STEP 8: Fusion (Pot + Plant)")
    print("=" * 60)

    # Load meshes
    pot_path = os.path.join(OUTPUT_DIR, "pot_mesh.ply")
    plant_path = os.path.join(OUTPUT_DIR, "plant_mesh.ply")

    if not os.path.exists(pot_path):
        print(f"ERROR: {pot_path} not found. Run step 4 first.")
        return
    if not os.path.exists(plant_path):
        print(f"ERROR: {plant_path} not found. Run step 7 first.")
        return

    print(f"Loading pot mesh: {pot_path}")
    pot_mesh = o3d.io.read_triangle_mesh(pot_path)
    print(f"  {len(pot_mesh.vertices)} verts, {len(pot_mesh.triangles)} tris")

    print(f"Loading plant mesh: {plant_path}")
    plant_mesh = o3d.io.read_triangle_mesh(plant_path)
    print(f"  {len(plant_mesh.vertices)} verts, {len(plant_mesh.triangles)} tris")

    # Merge
    print("\nMerging meshes...")
    fused = pot_mesh + plant_mesh
    print(f"  Before cleanup: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # Remove duplicate vertices
    fused.remove_duplicated_vertices()
    print(f"  After dedup: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # Remove degenerate triangles
    fused.remove_degenerate_triangles()
    print(f"  After degenerate removal: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # Remove non-finite vertices
    verts = np.asarray(fused.vertices)
    fin = np.all(np.isfinite(verts), axis=1)
    if not fin.all():
        fused.remove_vertices_by_mask(~fin)
        print(f"  After NaN removal: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # Taubin smooth x3
    print(f"\nTaubin smooth x{TAUBIN_ITERATIONS}...")
    fused = fused.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERATIONS)
    fused.compute_vertex_normals()

    # Save
    out_ply = os.path.join(OUTPUT_DIR, "fused_mesh.ply")
    out_obj = os.path.join(OUTPUT_DIR, "fused_mesh.obj")

    save_mesh(out_ply, fused)
    print(f"Saved: {out_ply}")

    o3d.io.write_triangle_mesh(out_obj, fused)
    print(f"Saved: {out_obj}")

    # Report
    report = {
        "pipeline": "8-step OpenCV registration pipeline",
        "date": "2026-06-01",
        "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
        "deg_per_frame": DEG_PER_FRAME,
        "num_frames": NUM_FRAMES,
        "pot_mesh": {
            "vertices": len(pot_mesh.vertices),
            "triangles": len(pot_mesh.triangles)
        },
        "plant_mesh": {
            "vertices": len(plant_mesh.vertices),
            "triangles": len(plant_mesh.triangles)
        },
        "fused_mesh": {
            "vertices": len(fused.vertices),
            "triangles": len(fused.triangles)
        },
        "outputs": [
            "fused_mesh.ply",
            "fused_mesh.obj",
            "pot_mesh.ply",
            "plant_mesh.ply"
        ]
    }
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification: Fused model — close window to finish")
    print("=" * 60)
    print("  Pot and plant should align, no gaps, no outliers")
    print("  Leaf detail should be preserved")
    visualize_mesh(fused, "Step 8 — Fused Model (Pot + Plant)")

    print("\n" + "=" * 60)
    print("8-STEP PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Outputs in {OUTPUT_DIR}/:")
    print("  fused_mesh.ply  — Final fused model")
    print("  fused_mesh.obj  — OBJ format")
    print("  pot_mesh.ply    — Pot-only mesh")
    print("  plant_mesh.ply  — Plant-only mesh")
    print("  report.json     — Reconstruction report")
    print("=" * 60)


if __name__ == "__main__":
    main()
