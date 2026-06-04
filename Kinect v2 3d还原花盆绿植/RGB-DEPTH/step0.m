

clc; clear all;
%% ================== 彩色+深度同步手动拍摄 ==================
% ========== 1. 可配置参数 ==========
total_angles = 75;
save_dir = 'kinect_sync_data_manual';
color_format = 'tif';

% ========== 2. 初始化环境 ==========
clear vid_color vid_depth;

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

vid_color = videoinput('kinect', 1);
vid_depth = videoinput('kinect', 2);

% ========== 3. 配置触发参数 ==========
vid_color.FramesPerTrigger = 1;
vid_depth.FramesPerTrigger = 1;
triggerconfig([vid_color vid_depth], 'manual');

fprintf('设备初始化完成，准备开始手动拍摄...\n');
pause(1);

% ========== 4. 手动循环拍摄 ==========
fprintf('=== 手动同步拍摄，共%d组 ===\n', total_angles);
fprintf('每一组拍摄前，请将转台转到目标角度，然后按 Enter 键触发拍摄。\n');
fprintf('如需中途停止，按 Ctrl+C 即可。\n\n');

for i = 1:total_angles
    % ---------- 等待用户按键 ----------
    fprintf('>>> 第%d/%d组：调整好角度后按 Enter 键触发拍摄', i, total_angles);
    pause;  % 等待用户按任意键（Enter）

    % ---------- 启动设备 ----------
    start([vid_color vid_depth]);
    pause(0.3);  % 给设备足够时间进入 Running

    % ---------- 同步触发与采集 ----------
    trigger([vid_color vid_depth]);
    [frame_color, ts_color, meta_color] = getdata(vid_color);
    [frame_depth, ts_depth, meta_depth] = getdata(vid_depth);

    % ---------- 停止设备 ----------
    stop([vid_color vid_depth]);

    % ---------- 数据保存 ----------
    file_idx = sprintf('%04d', i);

    color_path = fullfile(save_dir, [file_idx '_color.' color_format]);
    imwrite(frame_color, color_path);

    depth_path = fullfile(save_dir, [file_idx '_depth.tif']);
    imwrite(uint16(frame_depth), depth_path);

    mat_path = fullfile(save_dir, [file_idx '_meta.mat']);
    save(mat_path, 'ts_color', 'ts_depth', 'meta_color', 'meta_depth');

    fprintf('✓ 第%d/%d组保存完成\n\n', i, total_angles);
end

% ========== 5. 清理 ==========
imaqreset;
delete(imaqfind);
clear vid_color vid_depth;
fprintf('\n=== 全部拍摄完成！数据已保存至：%s ===\n', save_dir);