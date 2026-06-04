# Kinect v2 转台三维重建 — MATLAB→Python 迁移设计

日期：2026-05-29
来源规格：`kinect_reconstruction_guide.docx`
原始管线：`reconstruct_kinect_final.m`
产出脚本：`reconstruct_kinect_python.py`

## 目标

将 MATLAB 转台重建管线迁移到 Python + Open3D，并落实 guide 的三项改进：

1. 手动标注旋转中心（保留交互式选点）
2. 逐帧 Point-to-Plane ICP 精配准
3. 收紧绿叶颜色修复（KNN 加 2.5 cm 距离上限）

数据：Kinect v2 RGB-D × 25 帧，等间距 360° 转台（每帧 14.4°）。

## 脚本结构（对应 guide §3.1）

单文件，顶部常量区集中放置 guide §11 的全部参数。函数：

| 函数 | 职责 |
|------|------|
| `load_frame(i)` | 读 `{i:04d}_color.tif`/`_depth.tif` → scipy 中值滤波(3×3) → Kinect 反投影 → 相机系点+色 |
| `extract_roi(pts, cols, roi)` | 主体 / 严格 / 底座 ROI 裁剪（numpy 布尔掩码） |
| `estimate_center_auto(base_pts)` | 代数圆拟合 + XZ 网格搜索（移植 `fitCircleDiagnostic` + `searchCenterGrid` 紧致度评分） |
| `pick_center_interactive(pcd)` | Open3D `VisualizerWithEditing`，Shift+左键选点取平均 |
| `rotate_around_y(pts, center, theta)` | 绕 Y 轴旋转叠加 |
| `icp_refine(src, tgt)` | Point-to-Plane ICP，`ICP_MAX_DIST=0.02` |
| `repair_plant_colors(pcd)` | HSV 筛绿/白灰 + KDTreeFlann KNN，新增 2.5 cm 上限，仅处理 Y<0.02 |
| `save_preview(pcd, path)` | 离屏截图（best-effort，失败不影响 .ply 产出） |

## 坐标系与 ROI（沿用 MATLAB）

- 相机系：X 向右，Y 向下，Z 向前；旋转中心固定 Y=0，仅在 XZ 平面估算
- 主体 ROI：x∈[-0.20,0.20], y∈[-0.17,0.27], z∈[0.48,0.75]
- 严格 ROI：x∈[-0.15,0.15], y∈[-0.10,0.35], z∈[0.30,0.80]
- 底座/中心 ROI：x∈[-0.20,0.20], y∈[0.02,0.22], z∈[0.48,0.75]，每帧≤8000 点

## 旋转中心处理

- 默认提示 `[Open interactive center picker? Y/n]`，默认 Y → 弹窗交互选点（保留供人工使用）
- 输入 `n` 跳过 → 用 `estimate_center_auto`
- 常量 `HARDCODE_CENTER`（默认 `None`）；填入 `[0.0220, 0.0, 0.5321]` 即跳过交互（guide §4.3）
- 自动估算/硬编码兜底值：`[0.0220, 0.0, 0.5321]`（= MATLAB `fallbackCenter`）
- **自动化无 GUI 验证运行**：喂入 `n`，走自动估算中心

## 关键参数（guide §11）

```
FX = FY = 365.456 ; CX = 254.878 ; CY = 205.395
DEPTH_SCALE = 0.001 ; DEPTH_W = 512 ; DEPTH_H = 424
TOTAL_VIEWS = 25 ; ANGLE_STEP = 360/25 = 14.4°
ROTATION_SIGNS = [+1, -1]
ICP_ENABLED = True ; ICP_MAX_DIST = 0.02
DENSE_VOXEL = 0.0025 ; CLEAN_VOXEL = 0.005 ; STRICT_VOXEL = 0.005
COLOR_REPAIR_MAX_DIST = 0.025
SOR: nb_neighbors=20, std_ratio=2.0
```

## 输出（严格按 guide §7 命名）

- `final_dense_{pos,neg}.ply` — 2.5 mm 体素，含颜色修复
- `final_clean_{pos,neg}.ply` — 5 mm 体素 + SOR 去噪 + 颜色修复
- `final_strict_{pos,neg}.ply` — 严格 ROI，5 mm 体素 + SOR + 颜色修复
- `final_preview_{dense,clean,strict}_pos.png` — 俯视预览截图

## 运行与验证

1. `pip install open3d`（清华镜像）到 D:\Python 3.9.13
2. `echo n | python reconstruct_kinect_python.py`
3. 验证：回读各 .ply 点数（与既有 MATLAB 输出量级对比）；查看预览截图确认无明显鬼影/飞点；离屏渲染失败时用 matplotlib 回退渲染以便目检

## 范围约束（YAGNI）

- 旋转角固定 14.4° 等步进，不读 `meta.mat`（与 guide/MATLAB 一致）
- 不修改既有 MATLAB 脚本与既有输出文件，仅新增 Python 脚本及其新命名输出

## 测试策略（TDD，仅纯函数）

无 GUI、无 Open3D 依赖即可测：

- `rotate_around_y`：合成点绕已知中心旋转 360° 应回到原点；旋转矩阵正交性
- `extract_roi`：构造跨边界点集，验证仅保留范围内点且颜色对应
- `estimate_center_auto`：合成一个绕已知中心的圆环点云，估算中心应接近真值
