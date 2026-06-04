"""
Interactive crop tool — draw rectangles on 3 projection views.
- Drag to draw rectangle on any subplot
- Press 'k' to enter select mode
- Press 'c' to confirm and print ranges
- Press 'r' to reset rectangles
Close window when done.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os

DATA = r"E:\花瓶3d"

# ---- Load data ----
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

# Clip depth for better view
depth_mask = (zv > 0.3) & (zv < 4.0)
xs = x[depth_mask]; ys = y[depth_mask]; zs = zv[depth_mask]

# Subsample for performance
N = min(20000, len(xs))
idx = np.random.choice(len(xs), N, replace=False)
xs, ys, zs = xs[idx], ys[idx], zs[idx]

# ---- State ----
rects = {}  # {(ax_idx): [xmin, xmax, ymin, ymax]}
selectors = {}  # {(ax_idx): RectangleSelector}
current_ax = None

def on_select(eclick, erelease):
    """Called when rectangle is drawn on an axis"""
    for ax_idx, ax in enumerate(axes):
        if ax == eclick.inaxes:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            rects[ax_idx] = [min(x1,x2), max(x1,x2), min(y1,y2), max(y1,y2)]
            print(f"  View {['FRONT','TOP','SIDE'][ax_idx]}: "
                  f"x=[{rects[ax_idx][0]:.3f}, {rects[ax_idx][1]:.3f}]  "
                  f"y=[{rects[ax_idx][2]:.3f}, {rects[ax_idx][3]:.3f}]")

def toggle_selector(event):
    """De/activate selectors"""
    for sel in selectors.values():
        sel.set_active(not sel.active)
    state = "ON" if list(selectors.values())[0].active else "OFF"
    print(f"Selection mode: {state}")

def confirm(event):
    """Print crop ranges from rectangles"""
    if event.key != 'c':
        return
    if not rects:
        print("No rectangles drawn yet!")
        return

    print("\n" + "="*60)
    print("CROP RANGES (from your selections):")
    print("-"*60)

    # Front: X vs Y
    if 0 in rects:
        r = rects[0]
        print(f"  Front view (X vs Y):  X=[{r[0]:.3f}, {r[1]:.3f}]  Y=[{r[2]:.3f}, {r[3]:.3f}]")

    # Top: X vs Z
    if 1 in rects:
        r = rects[1]
        print(f"  Top view  (X vs Z):  X=[{r[0]:.3f}, {r[1]:.3f}]  Z=[{r[2]:.3f}, {r[3]:.3f}]")

    # Side: Z vs Y
    if 2 in rects:
        r = rects[2]
        print(f"  Side view (Z vs Y):  Z=[{r[0]:.3f}, {r[1]:.3f}]  Y=[{r[2]:.3f}, {r[3]:.3f}]")

    # Compute intersection
    x_ranges, y_ranges, z_ranges = [], [], []
    if 0 in rects:
        x_ranges.append(rects[0][:2])
        y_ranges.append(rects[0][2:])
    if 1 in rects:
        x_ranges.append(rects[1][:2])
        z_ranges.append(rects[1][2:])
    if 2 in rects:
        z_ranges.append(rects[2][:2])
        y_ranges.append(rects[2][2:])

    print("-"*60)
    if x_ranges:
        x_lo = max(r[0] for r in x_ranges)
        x_hi = min(r[1] for r in x_ranges)
        print(f"  X: [{x_lo:.3f}, {x_hi:.3f}] m")
    if y_ranges:
        y_lo = max(r[0] for r in y_ranges)
        y_hi = min(r[1] for r in y_ranges)
        print(f"  Y: [{y_lo:.3f}, {y_hi:.3f}] m")
    if z_ranges:
        z_lo = max(r[0] for r in z_ranges)
        z_hi = min(r[1] for r in z_ranges)
        print(f"  Z: [{z_lo:.3f}, {z_hi:.3f}] m")

    print("="*60)
    print("Copy these ranges and tell me to proceed to Step 2.")

def reset(event):
    """Clear all rectangles"""
    if event.key != 'r':
        return
    rects.clear()
    for ax in axes:
        for patch in ax.patches[:]:
            patch.remove()
    fig.canvas.draw()
    print("Rectangles cleared.")

# ---- Create figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.canvas.manager.set_window_title("Frame 0 — Draw rectangles around vase+plant, press 'c' to confirm")

# Front view: X vs Y (color by Z)
sc0 = axes[0].scatter(xs, ys, c=zs, s=2, cmap='turbo', alpha=0.8)
axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
axes[0].set_title('FRONT view\nDraw X vs Y range around vase+plant')
axes[0].axhline(0, color='gray', ls='--', alpha=0.3)
axes[0].axvline(0, color='gray', ls='--', alpha=0.3)
axes[0].set_aspect('equal')
plt.colorbar(sc0, ax=axes[0], label='Z depth (m)', shrink=0.8)

# Top view: X vs Z (color by Y)
sc1 = axes[1].scatter(xs, zs, c=ys, s=2, cmap='turbo', alpha=0.8)
axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Z (m)')
axes[1].set_title('TOP view\nDraw X vs Z range')
axes[1].axhline(0, color='gray', ls='--', alpha=0.3)
axes[1].axvline(0, color='gray', ls='--', alpha=0.3)
axes[1].set_aspect('equal')
plt.colorbar(sc1, ax=axes[1], label='Y height (m)', shrink=0.8)

# Side view: Z vs Y (color by X)
sc2 = axes[2].scatter(zs, ys, c=xs, s=2, cmap='turbo', alpha=0.8)
axes[2].set_xlabel('Z (m)'); axes[2].set_ylabel('Y (m)')
axes[2].set_title('SIDE view\nDraw Z vs Y range')
axes[2].axhline(0, color='gray', ls='--', alpha=0.3)
axes[2].axvline(0, color='gray', ls='--', alpha=0.3)
axes[2].set_aspect('equal')
plt.colorbar(sc2, ax=axes[2], label='X (m)', shrink=0.8)

plt.suptitle('INSTRUCTIONS: Drag to draw rectangle on each view |  k=toggle select |  c=confirm  |  r=reset', fontsize=14)
plt.tight_layout()

# Create rect selectors
for i, ax in enumerate(axes):
    sel = RectangleSelector(ax, on_select,
        useblit=True, button=[1],  # left mouse
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True,
        props=dict(facecolor='red', edgecolor='red', alpha=0.2, fill=True))
    selectors[i] = sel
    sel.set_active(False)  # start inactive

# Connect keys
fig.canvas.mpl_connect('key_press_event', toggle_selector)
fig.canvas.mpl_connect('key_press_event', confirm)
fig.canvas.mpl_connect('key_press_event', reset)

# First 'k' press activates
toggle_selector(None)

print("="*60)
print("CROP TOOL READY")
print("  Press 'k' to toggle selection on/off")
print("  When ON: drag on any view to draw rectangle")
print("  Press 'c' to confirm and print crop ranges")
print("  Press 'r' to reset all rectangles")
print("  Close window to exit")
print("="*60)

plt.show()
print("\nDone. Tell me your crop ranges to proceed.")
