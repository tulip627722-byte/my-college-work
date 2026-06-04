"""Generate 2D projections of frame 0 for manual crop estimation"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA = r"E:\花瓶3d"
OUT = os.path.join(DATA, "output")

d_raw = np.fromfile(os.path.join(DATA, "depth", "000.png"), dtype=np.uint8)
depth = cv2.imdecode(d_raw, cv2.IMREAD_UNCHANGED)
h, w = depth.shape

fx = fy = 365.456
cx, cy = 254.878, 205.395

v, u = np.mgrid[0:h, 0:w]
z = depth.astype(np.float64) / 1000.0
valid = z > 0.001
zv = z[valid]; uv = u[valid]; vv = v[valid]
x = (uv - cx) * zv / fx
y = (vv - cy) * zv / fy

# Color
c_raw = np.fromfile(os.path.join(DATA, "color", "000.png"), dtype=np.uint8)
color_rgb = cv2.cvtColor(cv2.imdecode(c_raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
u_rgb = np.clip((uv * 1920 / 512).astype(int), 0, 1919)
v_rgb = np.clip((vv * 1080 / 424).astype(int), 0, 1079)
colors = color_rgb[v_rgb, u_rgb] / 255.0

# Subsample for speed
N = min(30000, len(x))
idx = np.random.choice(len(x), N, replace=False)
xs, ys, zs = x[idx], y[idx], z[idx]
cs = colors[idx]

# Clip for better visualization
mask = (zs > 0.3) & (zs < 4.0)
xs, ys, zs = xs[mask], ys[mask], zs[mask]
cs = cs[mask]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Front view: X vs Y (color by Z depth)
sc0 = axes[0].scatter(xs, ys, c=zs, s=1, cmap='turbo', alpha=0.7)
axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
axes[0].set_title('FRONT view (X vs Y)\nY=0 is ground plane?')
axes[0].axhline(0, color='gray', ls='--', alpha=0.3)
axes[0].axvline(0, color='gray', ls='--', alpha=0.3)
axes[0].set_aspect('equal')
plt.colorbar(sc0, ax=axes[0], label='Z depth (m)', shrink=0.7)

# Top view: X vs Z (looking down)
sc1 = axes[1].scatter(xs, zs, c=ys, s=1, cmap='turbo', alpha=0.7)
axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Z (m, depth)')
axes[1].set_title('TOP view (X vs Z)')
axes[1].axhline(0, color='gray', ls='--', alpha=0.3)
axes[1].axvline(0, color='gray', ls='--', alpha=0.3)
axes[1].set_aspect('equal')
plt.colorbar(sc1, ax=axes[1], label='Y height (m)', shrink=0.7)

# Side view: Z vs Y (looking from side)
sc2 = axes[2].scatter(zs, ys, c=xs, s=1, cmap='turbo', alpha=0.7)
axes[2].set_xlabel('Z (m, depth)'); axes[2].set_ylabel('Y (m)')
axes[2].set_title('SIDE view (Z vs Y)')
axes[2].axhline(0, color='gray', ls='--', alpha=0.3)
axes[2].axvline(0, color='gray', ls='--', alpha=0.3)
axes[2].set_aspect('equal')
plt.colorbar(sc2, ax=axes[2], label='X (m)', shrink=0.7)

plt.suptitle(f'Frame 0 — {len(xs)} points shown\nUse these to pick crop ranges', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'frame0_projections.png'), dpi=150)
print(f"Saved: {OUT}/frame0_projections.png")

# Also print stats to help
print(f"\nX stats: min={x.min():.3f}  max={x.max():.3f}  mean={x.mean():.3f}")
print(f"Y stats: min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}")
print(f"Z stats: min={z[valid].min():.3f}  max={z[valid].max():.3f}  mean={z[valid].mean():.3f}")
print(f"\nLook at the PNG, then tell me your crop ranges like:")
print(f'  X: [min, max] m')
print(f'  Y: [min, max] m')
print(f'  Z: [min, max] m')
