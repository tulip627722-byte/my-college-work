%% ========== 基于离线图像的 Kinect V2 植物表型重构 ==========
clc; close all;

% 文件夹路径
data_path_rgb   = 'E:\RGB-DEPTH\RGB\jpg\';
data_path_depth = 'E:\RGB-DEPTH\DEEP\png\';

% ---------- 1. 自动获取文件列表并提取数字 ----------
rgb_list   = dir(fullfile(data_path_rgb, '*.jpg'));
depth_list = dir(fullfile(data_path_depth, '*.png'));

% 提取每个文件名中的数字部分（如 '1.jpg' → 1）
rgb_nums   = getFileNumbers({rgb_list.name});
depth_nums = getFileNumbers({depth_list.name});

% 找出两个文件夹共有的数字，并按数字顺序排序
[common_nums, ~] = intersect(rgb_nums, depth_nums);
common_nums = sort(common_nums);
num_views = length(common_nums);

if num_views == 0
    error('未找到任何匹配的图像对！请检查文件名是否包含相同数字。');
end
fprintf('共找到 %d 对匹配的图像（数字：%s）\n', num_views, mat2str(common_nums(:)'));

%% 2. 相机内参与外参（Kinect V2 典型值）
depth_intrinsics = cameraIntrinsics([365.0, 365.0], [256.0, 212.0], [512, 424]);
color_intrinsics = cameraIntrinsics([1060.0, 1060.0], [960.0, 540.0], [1920, 1080]);
R_d2c = eye(3);
T_d2c = [0.025, 0.0, 0.0];         % 米

% 滤波与分割参数
depth_range = [0.4, 1.2];           % 米
y_split = 0.02;                     % 米
angle_increment = 36;               % 每视角旋转角度（度）

%% 3. 循环处理配对的视角
plant_clouds = cell(1, num_views);
pot_clouds   = cell(1, num_views);
view_poses_plant = cell(1, num_views);

for idx = 1:num_views
    num = common_nums(idx);
    
    % 根据数字重新找到文件名（避免顺序依赖）
    rgb_name = sprintf('%d.jpg', num);
    depth_name = sprintf('%d.png', num);
    % 若实际文件名不是纯数字加扩展名，而是例如 "img_001.jpg" 则备用查找
    if ~exist(fullfile(data_path_rgb, rgb_name), 'file')
        % 从原始列表中查找包含该数字的文件
        match_rgb = find(rgb_nums == num, 1);
        rgb_name = rgb_list(match_rgb).name;
    end
    if ~exist(fullfile(data_path_depth, depth_name), 'file')
        match_depth = find(depth_nums == num, 1);
        depth_name = depth_list(match_depth).name;
    end
    
    color_file = fullfile(data_path_rgb, rgb_name);
    depth_file = fullfile(data_path_depth, depth_name);
    
    color_img = imread(color_file);
    depth_img = imread(depth_file);          % 16位PNG，单位毫米
    depth_m   = double(depth_img) / 1000.0; % 转米
    
    % 深度图 → 点云（无颜色）
    ptcloud_depth = depthToPointCloud(depth_m, depth_intrinsics);
    
    % 深度点云变换到彩色坐标系并投影提取颜色
    points_depth  = ptcloud_depth.Location;
    points_color  = (R_d2c * points_depth' + T_d2c(:))';
    uv = worldToImage(color_intrinsics, eye(3), zeros(1,3), points_color);
    u = round(uv(:,1));
    v = round(uv(:,2));
    
    valid = (u >= 1) & (u <= size(color_img,2)) & (v >= 1) & (v <= size(color_img,1));
    idx_valid = sub2ind(size(color_img, [1,2]), v(valid), u(valid));
    colors = zeros(size(points_depth,1), 3, 'uint8');
    for ch = 1:3
        channel = color_img(:,:,ch);
        colors(valid, ch) = channel(idx_valid);
    end
    
    ptcloud_full = pointCloud(points_depth(valid, :), 'Color', colors(valid, :));
    
    % 直通滤波
    z = ptcloud_full.Location(:,3);
    mask_z = (z >= depth_range(1)) & (z <= depth_range(2));
    ptcloud_pass = select(ptcloud_full, find(mask_z));
    
    % 统计离群点滤波
    ptcloud_filtered = pcdenoise(ptcloud_pass);
    
    % Y轴分割花盆与植株
    y_vals = ptcloud_filtered.Location(:,2);
    mask_plant = y_vals > y_split;
    mask_pot   = y_vals <= y_split;
    
    plant_clouds{idx} = select(ptcloud_filtered, find(mask_plant));
    pot_clouds{idx}   = select(ptcloud_filtered, find(mask_pot));
    
    % 视角逆旋转矩阵
    view_poses_plant{idx} = rotY(deg2rad(-angle_increment * (idx - 1)));
    
    fprintf('处理图像对 %d (%s / %s): 植株 %d 点, 花盆 %d 点\n', ...
        num, rgb_name, depth_name, plant_clouds{idx}.Count, pot_clouds{idx}.Count);
end

%% 4. 多视角对齐与融合
% 花盆圆拟合中心 (XZ平面)
pot_centers = zeros(num_views, 3);
for i = 1:num_views
    loc = pot_clouds{i}.Location;
    xz = loc(:, [1,3]);
    A = [2*xz(:,1), 2*xz(:,2), ones(size(xz,1),1)];
    B = xz(:,1).^2 + xz(:,2).^2;
    sol = A \ B;
    pot_centers(i,:) = [sol(1), 0, sol(2)];
end

ref_center = pot_centers(1,:);
aligned_plant = cell(1, num_views);
aligned_pot   = cell(1, num_views);

for i = 1:num_views
    T_pot = ref_center - pot_centers(i,:);
    tform_pot = rigid3d(eye(3), T_pot);
    
    aligned_pot{i} = pctransform(pot_clouds{i}, tform_pot);
    pt_trans = pctransform(plant_clouds{i}, tform_pot);
    R = view_poses_plant{i};
    loc_rot = (R * pt_trans.Location')';
    aligned_plant{i} = pointCloud(loc_rot, 'Color', pt_trans.Color);
end

merged_plant = pccat(aligned_plant{:});
merged_pot   = pccat(aligned_pot{:});
merged_full = pcmerge(merged_plant, merged_pot, 0.001);

%% 5. 颜色与密度优化
% HSV色彩修补
colors_rgb = merged_full.Color;
hsv = rgb2hsv(double(colors_rgb)/255);
white_mask = (hsv(:,2) < 0.1) & (hsv(:,3) > 0.9);
if any(white_mask)
    good_hue = hsv(~white_mask, 1);
    mean_hue = mean(good_hue);
    hsv(white_mask, 1) = mean_hue;
    hsv(white_mask, 2) = 0.3;
end
new_rgb = uint8(hsv2rgb(hsv) * 255);
merged_full = pointCloud(merged_full.Location, 'Color', new_rgb);

% 体素下采样
voxel_size = 0.002;
ptcloud_final = pcdownsample(merged_full, 'gridAverage', voxel_size);

%% 6. 保存结果
output_folder = 'E:\RGB-DEPTH\output\';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
pcwrite(ptcloud_final, fullfile(output_folder, 'maize_model.ply'), 'Encoding', 'binary');
save(fullfile(output_folder, 'maize_model.mat'), 'ptcloud_final', 'merged_full');
disp('处理完成！模型已保存至 E:\RGB-DEPTH\output\');

%% ================== 辅助函数 ==================
function ptcloud = depthToPointCloud(depth_m, intrinsics)
    [height, width] = size(depth_m);
    [u, v] = meshgrid(1:width, 1:height);
    u = u(:); v = v(:);
    z = depth_m(:);
    x = (u - intrinsics.PrincipalPoint(1)) .* z / intrinsics.FocalLength(1);
    y = (v - intrinsics.PrincipalPoint(2)) .* z / intrinsics.FocalLength(2);
    valid = (z > 0) & isfinite(z);
    ptcloud = pointCloud([x(valid), y(valid), z(valid)]);
end

function R = rotY(theta)
    c = cos(theta); s = sin(theta);
    R = [c, 0, s; 0, 1, 0; -s, 0, c];
end

function nums = getFileNumbers(filenames)
    % 从文件名中提取数字，例如 '1.jpg' → 1, 'img_002.png' → 2
    nums = zeros(size(filenames));
    for k = 1:length(filenames)
        num_str = regexp(filenames{k}, '\d+', 'match', 'once');
        if ~isempty(num_str)
            nums(k) = str2double(num_str);
        else
            nums(k) = NaN;
        end
    end
end