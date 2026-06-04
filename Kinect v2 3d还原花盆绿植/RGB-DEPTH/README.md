# Kinect V2 植物表型3D重建

## 项目概述

基于Kinect V2同步采集的25帧RGB-D数据，对盆栽植物进行高保真三维重建。处理流程包括时域深度滤波、点云生成、地面剔除、空间ROI隔离、绿叶/花盆语义分割，以及Poisson曲面重建。

**数据来源：** `E:/RGB-DEPTH/kinect_sync_data_manual/`（25组同步RGB-TIF + Depth-TIF）

**重建结果：**
- 原始点云：138,553 点 → 滤波后：28,026 点
- 绿叶：8,797 点 (31.4%) | 花盆：10,348 点 (36.9%)
- Poisson网格：52,402 顶点 / 103,818 三角面

---

## 处理流程

### 输入数据规格
| 通道 | 分辨率 | 格式 | 有效范围 |
|------|--------|------|----------|
| Color | 1920×1080 | uint8 RGB | 0–255 |
| Depth | 512×424 | uint16 (mm) | 400–2000 mm |

### Kinect V2 相机内参
| 参数 | 深度相机 | 彩色相机 |
|------|----------|----------|
| 焦距 fx/fy | 365.0 / 365.0 | 1080.0 / 1080.0 |
| 主点 cx/cy | 255.5 / 211.5 | 959.5 / 539.5 |
| 外参 (d→c) | — | R=I, t=[-0.052, 0, 0] m |

### 管线步骤

```
┌─────────────────┐
│ 1. 加载25帧      │ 读取 color.tif + depth.tif, 堆叠为depth_stack
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. 时域中值滤波   │ 逐像素median(depth_stack, 3), 抑制ToF随机噪声
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. 点云+着色     │ 深度→3D(depth cam) → 外参变换 → 投影到color平面采样
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. 滤波+ROI      │ 体素降采样(1mm) → 统计去噪 → 高度法地面剔除
│                 │ → 百分位空间ROI → 轻量去噪 → 小簇过滤
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. 语义分割      │ 自适应ExG(约束[5,18]) + 色相比 → 绿叶
│                 │ HSV色调 + 饱和度 + 几何约束 → 花盆
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. 曲面重建      │ pcnormals → 法线方向对齐 → Poisson重建(depth=8)
│                 │ → 退化面剔除 → Laplacian平滑
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. 结果输出      │ PLY点云/网格 + PNG可视化
└─────────────────┘
```

---

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `depth_min/max` | 0.4–2.0 m | 有效深度范围 |
| `voxel_size` | 0.001 m | 体素降采样粒度 |
| `stat_k / stat_std` | 12 / 1.5 | 统计滤波参数 |
| `floor_gap` | 0.05 m | 地面剔除距离缓冲 (Z < floor_mode - gap) |
| `exg_range` | [5, 18] | 自适应ExG阈值约束范围 |
| `min_cluster` | 50 点 | 最小连通簇过滤 |
| Poisson depth | 8 | Poisson重建八叉树深度 |
| Laplacian | 2 iter, λ=0.4 | 网格平滑 |

---

## 输出文件

生成至 `E:/RGB-DEPTH/results/`：

| 文件 | 大小 | 内容 |
|------|------|------|
| `cloud_raw.ply` | 2.0 MB | 原始点云 (138,553点, RGB) |
| `cloud_clean.ply` | 420 KB | 滤波+ROI后点云 (28,026点, RGB) |
| `cloud_leaf.ply` | 132 KB | 绿叶分割 (8,797点) |
| `cloud_pot.ply` | 155 KB | 花盆分割 (10,348点) |
| `cloud_plant.ply` | 287 KB | 植物+花盆合并 (19,145点) |
| `mesh_plant.ply` | 3.5 MB | Poisson曲面网格 (52,402顶点 / 103,818面) |
| `depth_filtered.tif` | 321 KB | 时序中值滤波深度图 |
| `pipeline_overview.png` | — | 处理流程6格对比 |
| `mesh_views.png` | — | 网格四角度视图 |
| `plant_pointcloud.png` | 462 KB | 植物着色点云 |

---

## 使用方法

### 环境要求
- **MATLAB R2025a** (含 Computer Vision Toolbox + Lidar Toolbox)
- 注意：R2025a中 `pcnormals` 返回法线矩阵（非pointCloud），`pc2surfacemesh` 需 `(ptCloud, "poisson")` 语法

### 运行

```matlab
>> run('E:/RGB-DEPTH/main_reconstruction.m')
```

或命令行（**使用软件OpenGL避免NVIDIA驱动崩溃**）：

```bash
matlab -softwareopengl -batch "run('E:/RGB-DEPTH/main_reconstruction.m')"
```

> 若GPU驱动导致 `saveas` 崩溃，可单独运行 `visualize_only.m` 生成PNG。

### 预期运行时间
- 25帧加载 + 时域滤波：~30秒
- 点云生成 + 滤波 + 分割：~60秒
- Poisson重建：~90秒
- 总计：约3分钟

---

## 算法说明

### 1. 时序中值滤波
对25帧每个深度像素取中值，利用真值稳定、噪声随机的原理抑制Kinect V2 ToF传感器的随机深度噪声，同时保留真实几何细节。

### 2. 高度法地面剔除
统计Z轴直方图，取最大频数bin为地面高度，保留 Z < floor_mode - 0.05m 的近景区域。相比RANSAC平面拟合，对非理想平面（如桌缘、杂物）更鲁棒。

### 3. 空间ROI
对非地面点云取X/Y/Z的5-95百分位作为边界，加10%边距，剔除边缘离散背景点。

### 4. 自适应超绿指数分割
ExG = 2G - R - B，对有效ExG值计算Otsu自适应阈值，约束在[5, 18]范围内。结合G>R+2且G>B+2的比率约束进行掩码膨胀与孔洞填充。

### 5. Poisson曲面重建
pcnormals估计点云法线 → 法线方向统一朝向相机 → pc2surfacemesh("poisson") 求解指示函数 → 退化三角面剔除 → Laplacian平滑。

---

## 效果评估

| 指标 | 评估 |
|------|------|
| 叶片覆盖 | 绿叶点8,797 (31.4%)，叶片区域分割完整 |
| 花盆还原 | 花盆点10,348 (36.9%)，盆体几何完整 |
| 网格密度 | 52,402顶点 / 103,818面，细节丰富 |
| 噪声控制 | 统计去噪 + 小簇过滤后飞点极少 |

---

## 目录结构

```
E:/RGB-DEPTH/
├── kinect_sync_data_manual/    # 原始25组RGB-D数据
├── results/                    # 全部重建输出
├── main_reconstruction.m       # 主处理管线
├── visualize_only.m            # 单独可视化脚本 (software OpenGL)
├── README.md                   # 本文档
└── diagnostic*.m               # 调试诊断脚本
```

---

*处理日期：2026-05-25 | 算法参考：基于Kinect V2的植物表型特征提取与重构*
