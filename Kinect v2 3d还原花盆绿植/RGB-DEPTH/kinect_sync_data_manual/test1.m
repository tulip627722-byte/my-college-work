% =========================================================================
% 1. 初始化环境与参数检测
% =========================================================================
numFrames = 25; % 25组数据
gridSize = 0.01; % 降采样栅格大小 (1cm)

% 创建点云视图集对象（属于 Computer Vision Toolbox）
vSet = pcviewset;

% 自动探测当前 MATLAB 版本的变换矩阵类型 (兼容 R2020a - R2026+)
[tmpTform, ~, ~] = pcregistericp(pclouds{1}, pclouds{1});
isNewVersion = isa(tmpTform, 'rigidtform3d');

% 初始化第一帧的绝对位姿（世界坐标系原点）
if isNewVersion
    absPose = rigidtform3d(); % 新版：预乘 R*X + t
else
    absPose = rigid3d();      % 旧版：后乘 X*R + t
end
vSet = addView(vSet, 1, absPose);

% =========================================================================
% 2. 顺序帧配准与位姿图构建
% =========================================================================
fprintf('开始顺序帧配准...\n');
for i = 1:(numFrames-1)
    % 提取相邻帧并前置降采样（减少噪点，大幅提升 ICP 速度）
    fixedCloud  = pcdownsample(pclouds{i}, 'gridAverage', gridSize);
    movingCloud = pcdownsample(pclouds{i+1}, 'gridAverage', gridSize);
    
    % ICP 精配准：计算 moving 到 fixed 的相对变换
    % 建议使用 'pointToPlane'（点到平面），对植物叶片曲面拟合效果最好
    [relTform, ~, rmse] = pcregistericp(movingCloud, fixedCloud, ...
        'Metric', 'pointToPlane', 'MaxDistance', 0.05);
    
    % 计算当前帧的绝对位姿 (累加变换)
    if isNewVersion
        % 新版乘法规律：A 矩阵前乘
        absPoseMat = relTform.A * absPose.A;
        absPose = rigidtform3d(absPoseMat);
    else
        % 旧版乘法规律：T 矩阵后乘
        absPoseMat = absPose.T * relTform.T;
        absPose = rigid3d(absPoseMat);
    end
    
    % 将当前帧的绝对位姿以及与前一帧的相对约束加入视图集
    vSet = addView(vSet, i+1, absPose);
    vSet = addConnection(vSet, i, i+1, relTform);
end

% =========================================================================
% 3. 闭环检测 (Loop Closure) —— 解决 25 帧累积误差的关键
% =========================================================================
fprintf('执行首尾闭环检测...\n');
% 让最后一帧（第25帧）与第一帧进行 ICP 匹配
fixedRoot  = pcdownsample(pclouds{1}, 'gridAverage', gridSize);
movingEnd  = pcdownsample(pclouds{numFrames}, 'gridAverage', gridSize);

[loopTform, ~, loopRmse] = pcregistericp(movingEnd, fixedRoot, ...
    'Metric', 'pointToPlane', 'MaxDistance', 0.05);

% 如果首尾匹配的均方根误差在可接受范围内，则添加闭环约束
if loopRmse < 0.03 
    vSet = addConnection(vSet, numFrames, 1, loopTform);
    fprintf('成功检测到闭环，RMSE: %.4f\n', loopRmse);
else
    warning('闭环优化未触发：首尾帧重叠度不足或误差过大。');
end

% =========================================================================
% 4. 全局位姿图优化 (Pose Graph Optimization)
% =========================================================================
fprintf('正在进行全局位姿优化以均摊误差...\n');
vSetOptimized = optimizePoses(vSet);

% =========================================================================
% 5. 提取优化到位姿并重新融合点云
% =========================================================================
fprintf('正在根据优化到位姿重新拼接点云...\n');
% 获取优化后的所有帧绝对位姿表格
posesTable = vSetOptimized.Views;

% 以第一帧为基础构建全局点云
ptCloudGlobal = pclouds{1}; 

for i = 2:numFrames
    % 从表格中提取第 i 帧被校正后的绝对位姿
    if isNewVersion
        correctedPose = posesTable.AbsolutePose(i);
    else
        correctedPose = posesTable.AbsolutePose{i}; % 旧版可能是 cell 存储
    end
    
    % 变换原始点云
    alignedCloud = pctransform(pclouds{i}, correctedPose);
    
    % 融合点云，0.002 (2mm) 栅格用于去重合并，防止点云过密
    ptCloudGlobal = pcmerge(ptCloudGlobal, alignedCloud, 0.002);
end

% =========================================================================
% 6. 可视化结果
% =========================================================================
figure('Color', [0 0 0]); % 黑色背景更利于观察点云
pcshow(ptCloudGlobal);
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Computer Vision Toolbox 全局优化后的拼接结果', 'Color', 'w');