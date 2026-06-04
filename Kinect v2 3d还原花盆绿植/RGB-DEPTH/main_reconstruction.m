%% Kinect V2 Turntable Point-Cloud Reconstruction (Target Style)
% Point-cloud-first pipeline for turntable captures.
% Goal: dense colored point cloud with pot + tray + base retained.

clear; close all; clc;

%% ==================== 1) Configuration ====================
DATA_DIR = 'E:/RGB-DEPTH/kinect_sync_data_manual';
OUTPUT_DIR = 'E:/RGB-DEPTH/results';
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% Kinect V2 depth camera intrinsics (512x424)
fx_d = 365.0; fy_d = 365.0;
cx_d = 255.5; cy_d = 211.5;

% Kinect V2 color camera intrinsics (1920x1080)
fx_c = 1080.0; fy_c = 1080.0;
cx_c = 959.5;  cy_c = 539.5;

% Depth-to-color extrinsics (approx.)
R_d2c = eye(3);
t_d2c = [-0.052; 0; 0];

params = struct();

% Foreground extraction per frame
params.depth_min_m = 0.32;
params.depth_max_m = 2.00;
params.center_crop_ratio = 0.94;
params.depth_median_kernel = [3 3];
params.min_depth_pixels = 2500;
params.frame_voxel = 0.0018;
params.frame_stat_k = 10;
params.frame_stat_std = 1.6;
params.frame_cluster_eps = 0.018;
params.frame_cluster_min = 140;
params.min_points_per_frame = 2200;

% Pairwise ICP (adjacent + cumulative)
params.icp_coarse_grid = 0.006;
params.icp_fine_grid = 0.003;
params.icp_coarse_metric = 'pointToPoint';
params.icp_fine_metric = 'pointToPlaneWithColor';
params.icp_coarse_max_iter = 70;
params.icp_fine_max_iter = 90;
params.icp_coarse_tol = [1e-4, 0.07];
params.icp_fine_tol = [1e-5, 0.02];
params.icp_coarse_inlier_dist = 0.035;
params.icp_fine_inlier_dist = 0.014;
params.icp_pair_rmse_max = 0.018;
params.icp_pair_overlap_min = 0.38;

% Global look-back refinement
params.global_reg_grid = 0.0035;
params.global_rmse_max = 0.013;
params.global_overlap_min = 0.50;
params.global_fallback_overlap = 0.46;
params.global_max_points = 200000;   % lower threshold → downsample earlier → less peak RAM
params.global_compact_grid = 0.0022; % slightly coarser grid to keep map lean

% Fusion + cleanup
params.merge_grid = 0.0014;
params.final_voxel = 0.0015;
params.final_stat_k = 14;
params.final_stat_std = 1.35;
params.final_cluster_eps = 0.015;
params.final_cluster_min = 520;
params.overlap_radius = 0.008;
params.percentile_trim = [0.15, 99.85];

% Optional mesh
params.mesh_enable = true;
params.mesh_max_points = 45000;

% Final visualization
params.render_target_file = fullfile(OUTPUT_DIR, 'final_target_style.png');
params.render_view = [-31, 18];
params.render_marker = 7;
params.render_max_points = 85000;

fprintf('=== Turntable Point-Cloud Reconstruction (Target Style) ===\n');
fprintf('Data dir: %s\n', DATA_DIR);

%% ==================== 2) Read file list ====================
[depth_files, color_files] = listRgbdFiles(DATA_DIR);
num_frames = min(numel(depth_files), numel(color_files));
if num_frames < 3
    error('Need at least 3 RGB-D frame pairs; found %d.', num_frames);
end
fprintf('Frames found: %d\n', num_frames);

%% ==================== 3) Build per-frame clouds ====================
fprintf('\n[1/8] Building foreground clouds per frame...\n');
frame_clouds = cell(num_frames, 1);
raw_point_counts = zeros(num_frames, 1);

for i = 1:num_frames
    depth_img = imread(fullfile(DATA_DIR, depth_files{i}));
    color_img = imread(fullfile(DATA_DIR, color_files{i}));

    pt = createFrameCloud(depth_img, color_img, fx_d, fy_d, cx_d, cy_d, ...
        fx_c, fy_c, cx_c, cy_c, R_d2c, t_d2c, params);

    if isempty(pt)
        fprintf('  Frame %02d: skipped (no valid cloud).\n', i);
        continue;
    end

    raw_point_counts(i) = pt.Count;

    pt = preprocessFrameCloud(pt, params);

    if pt.Count < params.min_points_per_frame
        fprintf('  Frame %02d: skipped after cleanup (%d pts).\n', i, pt.Count);
        continue;
    end

    frame_clouds{i} = pt;

    if mod(i, 5) == 0 || i == num_frames
        fprintf('  Processed %d/%d\n', i, num_frames);
    end
end

valid_mask = ~cellfun(@isempty, frame_clouds);
valid_idx = find(valid_mask);
frame_clouds = frame_clouds(valid_mask);
raw_point_counts = raw_point_counts(valid_mask);
num_valid = numel(frame_clouds);

if num_valid < 3
    error('Too few usable frames (%d).', num_valid);
end
fprintf('Usable frames: %d/%d\n', num_valid, num_frames);

%% ==================== 4) Stage-1 registration ====================
% Adjacent registration with turntable-angle initial transform + ICP refinement.
fprintf('\n[2/8] Stage-1 adjacent registration...\n');

% --- Turntable geometry ---
% 25 frames, 360 degrees total => 14.4 deg per step.
% Rotation axis is Y (vertical). Try both CW and CCW; pick by RMSE on frame 2.
deg_per_step = 360.0 / num_valid;

% Build Ry(theta) helper: rotation about world Y axis
make_Ry = @(deg) [cosd(deg) 0 sind(deg) 0; ...
                  0          1 0          0; ...
                 -sind(deg) 0 cosd(deg) 0; ...
                  0          0 0          1];

% Test CW (+) vs CCW (-) on the first adjacent pair
init_cw  = rigidtform3d(make_Ry( deg_per_step));
init_ccw = rigidtform3d(make_Ry(-deg_per_step));
[~, st_cw,  ~] = registerWithFallback(frame_clouds{2}, frame_clouds{1}, params, init_cw);
[~, st_ccw, ~] = registerWithFallback(frame_clouds{2}, frame_clouds{1}, params, init_ccw);
if st_cw.rmse <= st_ccw.rmse
    sign_rot = +1;
    fprintf('  Turntable direction: clockwise (+%.1f deg/step)\n', deg_per_step);
else
    sign_rot = -1;
    fprintf('  Turntable direction: counter-clockwise (-%.1f deg/step)\n', deg_per_step);
end

pair_tforms = cell(num_valid, 1);
pair_tforms{1} = rigidtform3d;
pair_rmse = nan(num_valid, 1);
pair_overlap = nan(num_valid, 1);
pair_accepted = false(num_valid, 1);

pair_accepted(1) = true;
pair_rmse(1) = 0;
pair_overlap(1) = 1;

for k = 2:num_valid
    moving = frame_clouds{k};
    fixed = frame_clouds{k - 1};

    % Initial guess: one turntable step rotation about Y
    init_tform = rigidtform3d(make_Ry(sign_rot * deg_per_step));

    [~, st, pair_t] = registerWithFallback(moving, fixed, params, init_tform);

    if st.ok && st.overlap >= params.icp_pair_overlap_min && st.rmse <= 0.10
        pair_tforms{k} = pair_t;
        pair_accepted(k) = true;
        pair_rmse(k) = st.rmse;
        pair_overlap(k) = st.overlap;
        fprintf('  Pair %02d->%02d accepted | RMSE=%.4f | overlap=%.3f\n', ...
            valid_idx(k), valid_idx(k-1), st.rmse, st.overlap);
    else
        % Fallback: use pure turntable transform without ICP
        pair_tforms{k} = init_tform;
        pair_accepted(k) = true;
        pair_rmse(k) = st.rmse;
        pair_overlap(k) = st.overlap;
        fprintf('  Pair %02d->%02d fallback-turntable | RMSE=%.4f | overlap=%.3f\n', ...
            valid_idx(k), valid_idx(k-1), st.rmse, st.overlap);
    end
end

% Accumulate transforms: every frame now has a valid pair_tform (ICP or turntable fallback)
aligned_stage1 = cell(num_valid, 1);
aligned_stage1{1} = frame_clouds{1};
for k = 2:num_valid
    temp = frame_clouds{k};
    for j = k:-1:2
        temp = pctransform(temp, pair_tforms{j});
    end
    aligned_stage1{k} = temp;
end

stage1_mask = ~cellfun(@isempty, aligned_stage1);
stage1_mask(1) = true;
aligned_stage1 = aligned_stage1(stage1_mask);
stage1_idx = valid_idx(stage1_mask);
stage1_rmse = pair_rmse(stage1_mask);
stage1_overlap = pair_overlap(stage1_mask);

if numel(aligned_stage1) < 3
    error('Stage-1 accepted fewer than 3 frames.');
end

fprintf('Stage-1 accepted: %d/%d\n', numel(aligned_stage1), num_valid);

%% ==================== 5) Stage-2 global look-back refinement ====================
% Refine each aligned frame against the current fused global map.
fprintf('\n[3/8] Stage-2 global look-back refinement...\n');

global_map = aligned_stage1{1};
refined_frames = cell(numel(aligned_stage1), 1);
refined_frames{1} = aligned_stage1{1};

global_rmse = nan(numel(aligned_stage1), 1);
global_overlap = nan(numel(aligned_stage1), 1);
global_keep = false(numel(aligned_stage1), 1);

global_rmse(1) = 0;
global_overlap(1) = 1;
global_keep(1) = true;

for k = 2:numel(aligned_stage1)
    moving = aligned_stage1{k};
    fixed_for_reg = pcdownsample(global_map, 'gridAverage', params.global_reg_grid);

    [moving_refined, st] = registerWithFallback(moving, fixed_for_reg, params, []);

    if st.ok && st.rmse <= params.global_rmse_max && st.overlap >= params.global_overlap_min
        refined_frames{k} = moving_refined;
        global_map = pcmerge(global_map, moving_refined, params.merge_grid);
        if global_map.Count > params.global_max_points
            global_map = pcdownsample(global_map, 'gridAverage', params.global_compact_grid);
        end
        global_keep(k) = true;
        global_rmse(k) = st.rmse;
        global_overlap(k) = st.overlap;

        fprintf('  Frame %02d global-refined | RMSE=%.4f | overlap=%.3f | merged=%d\n', ...
            stage1_idx(k), st.rmse, st.overlap, global_map.Count);
    else
        % Fallback: keep stage-1 alignment only if overlap is acceptable.
        ov_fb = computeOverlapRatio(moving, fixed_for_reg, params.overlap_radius);
        if ov_fb >= params.global_fallback_overlap
            refined_frames{k} = moving;
            global_map = pcmerge(global_map, moving, params.merge_grid);
            if global_map.Count > params.global_max_points
                global_map = pcdownsample(global_map, 'gridAverage', params.global_compact_grid);
            end
            global_keep(k) = true;
            global_rmse(k) = st.rmse;
            global_overlap(k) = ov_fb;

            fprintf('  Frame %02d fallback-merged | RMSE=%.4f | overlap=%.3f | merged=%d\n', ...
                stage1_idx(k), st.rmse, ov_fb, global_map.Count);
        else
            fprintf('  Frame %02d skipped in stage-2 | RMSE=%.4f | overlap=%.3f\n', ...
                stage1_idx(k), st.rmse, st.overlap);
        end
    end
end

refined_frames = refined_frames(global_keep);
refined_idx = stage1_idx(global_keep);
refined_rmse = global_rmse(global_keep);
refined_overlap = global_overlap(global_keep);

valid_ratio = numel(refined_frames) / num_frames;
fprintf('Stage-2 kept: %d/%d (%.1f%% of all frames)\n', ...
    numel(refined_frames), num_frames, 100 * valid_ratio);

if isempty(refined_frames)
    error('No frame survived stage-2.');
end

%% ==================== 6) Fusion cleanup ====================
fprintf('\n[4/8] Final fusion cleanup...\n');

ptCloud_raw = global_map;

ptCloud_clean = pcdownsample(ptCloud_raw, 'gridAverage', params.final_voxel);
ptCloud_clean = pcdenoise(ptCloud_clean, 'NumNeighbors', params.final_stat_k, 'Threshold', params.final_stat_std);
ptCloud_clean = keepLargestCluster(ptCloud_clean, params.final_cluster_eps, params.final_cluster_min);
ptCloud_clean = trimByPercentile(ptCloud_clean, params.percentile_trim);

fprintf('  Raw fused cloud : %d points\n', ptCloud_raw.Count);
fprintf('  Clean fused cloud: %d points\n', ptCloud_clean.Count);

% Compatibility output: keep cloud_plant as full retained foreground.
ptCloud_plant = ptCloud_clean;

% Compatibility output: lightweight semantic split.
[ptCloud_leaf, ptCloud_pot] = deriveCompatSegments(ptCloud_clean);

fprintf('  Leaf compat cloud: %d points\n', ptCloud_leaf.Count);
fprintf('  Pot  compat cloud: %d points\n', ptCloud_pot.Count);

%% ==================== 7) Optional mesh ====================
fprintf('\n[5/8] Optional mesh generation...\n');
mesh = [];
if params.mesh_enable && ptCloud_clean.Count > 1200
    mesh = tryBuildOptionalMesh(ptCloud_clean, params);
end
if isempty(mesh)
    fprintf('  Mesh skipped/failed (point-cloud output remains valid).\n');
else
    fprintf('  Mesh generated: %d vertices, %d faces\n', mesh.NumVertices, mesh.NumFaces);
end

%% ==================== 8) Save outputs + render ====================
fprintf('\n[6/8] Saving outputs...\n');

pcwrite(ptCloud_raw, fullfile(OUTPUT_DIR, 'cloud_raw.ply'), 'PLYFormat', 'binary');
pcwrite(ptCloud_clean, fullfile(OUTPUT_DIR, 'cloud_clean.ply'), 'PLYFormat', 'binary');
pcwrite(ptCloud_plant, fullfile(OUTPUT_DIR, 'cloud_plant.ply'), 'PLYFormat', 'binary');
pcwrite(ptCloud_leaf, fullfile(OUTPUT_DIR, 'cloud_leaf.ply'), 'PLYFormat', 'binary');
pcwrite(ptCloud_pot, fullfile(OUTPUT_DIR, 'cloud_pot.ply'), 'PLYFormat', 'binary');

if ~isempty(mesh) && mesh.NumFaces > 0
    writeSurfaceMesh(mesh, fullfile(OUTPUT_DIR, 'mesh_plant.ply'));
else
    pcwrite(ptCloud_plant, fullfile(OUTPUT_DIR, 'mesh_plant.ply'), 'PLYFormat', 'binary');
end

save(fullfile(OUTPUT_DIR, 'registration_metrics.mat'), ...
    'valid_idx', 'stage1_idx', 'refined_idx', ...
    'raw_point_counts', 'stage1_rmse', 'stage1_overlap', ...
    'refined_rmse', 'refined_overlap', 'valid_ratio');

fprintf('  Saved point-cloud files and registration_metrics.mat\n');

fprintf('\n[7/8] Rendering target-style figure...\n');
render_ok = renderTargetStyle(ptCloud_plant, params.render_target_file, params);
if render_ok
    fprintf('  Saved: %s\n', params.render_target_file);
else
    warning('Failed to export target-style image reliably.');
end

fprintf('\n[8/8] Summary\n');
fprintf('  Accepted frame ratio: %.1f%%\n', 100 * valid_ratio);
fprintf('  Stage-1 mean RMSE   : %.4f m\n', mean(stage1_rmse(stage1_rmse > 0), 'omitnan'));
fprintf('  Stage-2 mean RMSE   : %.4f m\n', mean(refined_rmse(refined_rmse > 0), 'omitnan'));
fprintf('  Output folder       : %s\n', OUTPUT_DIR);
fprintf('\n=== Done ===\n');

%% ==================== Helper functions ====================

function [depth_files, color_files] = listRgbdFiles(data_dir)
    depth_dir = dir(fullfile(data_dir, '*_depth.tif'));
    color_dir = dir(fullfile(data_dir, '*_color.tif'));

    if isempty(depth_dir) || isempty(color_dir)
        error('No RGB-D files found in %s', data_dir);
    end

    [~, idd] = sort({depth_dir.name});
    [~, idc] = sort({color_dir.name});

    depth_files = {depth_dir(idd).name};
    color_files = {color_dir(idc).name};
end

function ptCloud = createFrameCloud(depth_img, color_img, fx_d, fy_d, cx_d, cy_d, ...
    fx_c, fy_c, cx_c, cy_c, R_d2c, t_d2c, params)

    depth_mm = double(depth_img);
    depth_mm = medfilt2(depth_mm, params.depth_median_kernel, 'symmetric');

    [h, w] = size(depth_mm);
    [u_grid, v_grid] = meshgrid(1:w, 1:h);

    half_u = round((w * params.center_crop_ratio) / 2);
    half_v = round((h * params.center_crop_ratio) / 2);
    center_mask = abs(u_grid - cx_d) <= half_u & abs(v_grid - cy_d) <= half_v;

    depth_valid = depth_mm > params.depth_min_m * 1000 & depth_mm < params.depth_max_m * 1000;
    valid = depth_valid & center_mask;

    if nnz(valid) < params.min_depth_pixels
        ptCloud = [];
        return;
    end

    u = u_grid(valid);
    v = v_grid(valid);
    z = depth_mm(valid) / 1000.0;

    X_d = (u - cx_d) .* z / fx_d;
    Y_d = (v - cy_d) .* z / fy_d;
    Z_d = z;

    pts_d = [X_d(:)'; Y_d(:)'; Z_d(:)'];
    pts_c = R_d2c * pts_d + t_d2c;

    u_c = (pts_c(1, :) ./ pts_c(3, :)) * fx_c + cx_c;
    v_c = (pts_c(2, :) ./ pts_c(3, :)) * fy_c + cy_c;

    img_h = size(color_img, 1);
    img_w = size(color_img, 2);
    good_proj = isfinite(u_c) & isfinite(v_c) & pts_c(3, :) > 0 & ...
        u_c >= 1 & u_c <= img_w & v_c >= 1 & v_c <= img_h;

    if nnz(good_proj) < params.min_points_per_frame
        ptCloud = [];
        return;
    end

    X_d = X_d(good_proj);
    Y_d = Y_d(good_proj);
    Z_d = Z_d(good_proj);

    u_px = round(u_c(good_proj));
    v_px = round(v_c(good_proj));
    lin_idx = sub2ind([img_h, img_w], v_px(:), u_px(:));

    colors = zeros(numel(lin_idx), 3, 'uint8');
    for ch = 1:3
        ch_img = color_img(:, :, ch);
        colors(:, ch) = ch_img(lin_idx);
    end

    ptCloud = pointCloud(single([X_d(:), Y_d(:), Z_d(:)]), 'Color', colors);
end

function ptOut = preprocessFrameCloud(ptIn, params)
    ptOut = pcdownsample(ptIn, 'gridAverage', params.frame_voxel);

    if ptOut.Count > 700
        ptOut = pcdenoise(ptOut, 'NumNeighbors', params.frame_stat_k, 'Threshold', params.frame_stat_std);
    end

    ptOut = keepLargestCluster(ptOut, params.frame_cluster_eps, params.frame_cluster_min);
end

function ptOut = keepLargestCluster(ptIn, epsilon, min_pts)
    if isempty(ptIn) || ptIn.Count < min_pts
        ptOut = ptIn;
        return;
    end

    % Fast path: use built-in Euclidean clustering.
    try
        [labels, num_clusters] = pcsegdist(ptIn, epsilon);
        if num_clusters > 0
            counts = zeros(num_clusters, 1);
            for c = 1:num_clusters
                counts(c) = sum(labels == c);
            end

            [best_count, best] = max(counts);
            if best_count >= min_pts
                ptOut = select(ptIn, labels == best);
            else
                ptOut = ptIn;
            end
            return;
        end
    catch
        % Fall through to lightweight fallback.
    end

    % Fallback: avoid very slow O(N^2) custom clustering on huge clouds.
    if ptIn.Count > 45000
        ptOut = ptIn;
        return;
    end

    pts = double(ptIn.Location);
    [labels, num_clusters] = dbscanImpl(pts, epsilon, min_pts);
    if num_clusters < 1
        ptOut = ptIn;
        return;
    end

    counts = zeros(num_clusters, 1);
    for c = 1:num_clusters
        counts(c) = sum(labels == c);
    end

    [~, best] = max(counts);
    keep = labels == best;
    ptOut = pointCloud(single(pts(keep, :)), 'Color', ptIn.Color(keep, :));
end

function [labels, num_clusters] = dbscanImpl(pts, epsilon, min_pts)
    n = size(pts, 1);
    labels = zeros(n, 1);
    cluster_id = 0;

    kdtree = KDTreeSearcher(pts);

    for i = 1:n
        if labels(i) ~= 0
            continue;
        end

        idx = rangesearch(kdtree, pts(i, :), epsilon);
        neighbors = idx{1};

        if numel(neighbors) < min_pts
            labels(i) = -1;
            continue;
        end

        cluster_id = cluster_id + 1;
        labels(i) = cluster_id;

        seed = neighbors;
        ptr = 1;

        while ptr <= numel(seed)
            j = seed(ptr);

            if labels(j) == -1
                labels(j) = cluster_id;
            end

            if labels(j) == 0
                labels(j) = cluster_id;
                idxj = rangesearch(kdtree, pts(j, :), epsilon);
                nbhj = idxj{1};
                if numel(nbhj) >= min_pts
                    seed = [seed, nbhj]; %#ok<AGROW>
                end
            end

            ptr = ptr + 1;
        end
    end

    num_clusters = cluster_id;
end

function [moving_aligned, st, t_total] = registerWithFallback(moving_full, fixed_full, params, initial_tform)
    moving_aligned = [];
    t_total = rigidtform3d;
    st = struct('ok', false, 'rmse', inf, 'rmse_coarse', inf, 'overlap', 0, 'metric', 'none');

    if isempty(moving_full) || isempty(fixed_full)
        return;
    end

    fixed_coarse = pcdownsample(fixed_full, 'gridAverage', params.icp_coarse_grid);
    moving_coarse = pcdownsample(moving_full, 'gridAverage', params.icp_coarse_grid);
    fixed_fine = pcdownsample(fixed_full, 'gridAverage', params.icp_fine_grid);
    moving_fine = pcdownsample(moving_full, 'gridAverage', params.icp_fine_grid);

    if fixed_coarse.Count < 500 || moving_coarse.Count < 500 || ...
       fixed_fine.Count < 700 || moving_fine.Count < 700
        return;
    end

    if nargin < 4 || isempty(initial_tform)
        init = rigidtform3d;
    else
        init = initial_tform;
    end

    try
        [t_coarse, ~, rmse_coarse] = pcregistericp(moving_coarse, fixed_coarse, ...
            'Metric', params.icp_coarse_metric, ...
            'MaxIterations', params.icp_coarse_max_iter, ...
            'Tolerance', params.icp_coarse_tol, ...
            'InlierDistance', params.icp_coarse_inlier_dist, ...
            'InitialTransform', init);
    catch
        try
            [t_coarse, ~, rmse_coarse] = pcregistericp(moving_coarse, fixed_coarse, ...
                'Metric', 'pointToPoint', ...
                'MaxIterations', params.icp_coarse_max_iter, ...
                'Tolerance', params.icp_coarse_tol, ...
                'InlierDistance', params.icp_coarse_inlier_dist);
        catch
            return;
        end
    end

    moving_fine_init = pctransform(moving_fine, t_coarse);

    try
        [t_fine, moving_fine_reg, rmse_fine] = pcregistericp(moving_fine_init, fixed_fine, ...
            'Metric', params.icp_fine_metric, ...
            'MaxIterations', params.icp_fine_max_iter, ...
            'Tolerance', params.icp_fine_tol, ...
            'InlierDistance', params.icp_fine_inlier_dist);
        metric_used = params.icp_fine_metric;
    catch
        try
            [t_fine, moving_fine_reg, rmse_fine] = pcregistericp(moving_fine_init, fixed_fine, ...
                'Metric', 'pointToPlane', ...
                'MaxIterations', params.icp_fine_max_iter, ...
                'Tolerance', params.icp_fine_tol, ...
                'InlierDistance', params.icp_fine_inlier_dist);
            metric_used = 'pointToPlane';
        catch
            try
                [t_fine, moving_fine_reg, rmse_fine] = pcregistericp(moving_fine_init, fixed_fine, ...
                    'Metric', 'pointToPoint', ...
                    'MaxIterations', params.icp_fine_max_iter, ...
                    'Tolerance', params.icp_fine_tol, ...
                    'InlierDistance', params.icp_fine_inlier_dist);
                metric_used = 'pointToPoint';
            catch
                return;
            end
        end
    end

    overlap = computeOverlapRatio(moving_fine_reg, fixed_fine, params.overlap_radius);

    moving_aligned = pctransform(pctransform(moving_full, t_coarse), t_fine);
    A_total = getTformMatrix(t_fine) * getTformMatrix(t_coarse);
    t_total = makeAffineTform(A_total);

    st.ok = true;
    st.rmse = rmse_fine;
    st.rmse_coarse = rmse_coarse;
    st.overlap = overlap;
    st.metric = metric_used;
end

function overlap = computeOverlapRatio(moving, fixed, radius)
    overlap = 0;
    if isempty(moving) || isempty(fixed) || moving.Count < 20 || fixed.Count < 20
        return;
    end

    % Cap point counts before building KDTree to avoid out-of-memory
    MAX_PTS = 20000;

    mpts = double(moving.Location);
    fpts = double(fixed.Location);

    if size(fpts, 1) > MAX_PTS
        step = floor(size(fpts, 1) / MAX_PTS);
        fpts = fpts(1:step:end, :);
    end

    if size(mpts, 1) > MAX_PTS
        step = floor(size(mpts, 1) / MAX_PTS);
        mpts = mpts(1:step:end, :);
    end

    kdt = KDTreeSearcher(fpts);
    idx = rangesearch(kdt, mpts, radius);
    matched = cellfun(@(c) ~isempty(c), idx);
    overlap = mean(matched);
end

function A = getTformMatrix(tform)
    A = eye(4);
    if isempty(tform)
        return;
    end

    try
        if isprop(tform, 'A')
            A = tform.A;
            return;
        end
    catch
    end

    try
        if isprop(tform, 'T')
            A = tform.T;
            return;
        end
    catch
    end
end

function tform = makeAffineTform(A)
    try
        tform = affinetform3d(A);
    catch
        tform = affine3d(A);
    end
end

function ptOut = trimByPercentile(ptIn, p_lim)
    if isempty(ptIn) || ptIn.Count < 100
        ptOut = ptIn;
        return;
    end

    pts = double(ptIn.Location);
    lo = prctile(pts, p_lim(1), 1);
    hi = prctile(pts, p_lim(2), 1);

    keep = pts(:,1) >= lo(1) & pts(:,1) <= hi(1) & ...
           pts(:,2) >= lo(2) & pts(:,2) <= hi(2) & ...
           pts(:,3) >= lo(3) & pts(:,3) <= hi(3);

    if nnz(keep) < 100
        ptOut = ptIn;
    else
        ptOut = pointCloud(single(pts(keep, :)), 'Color', ptIn.Color(keep, :));
    end
end

function [leafCloud, potCloud] = deriveCompatSegments(ptIn)
    if isempty(ptIn) || ptIn.Count < 50
        leafCloud = ptIn;
        potCloud = ptIn;
        return;
    end

    cols = double(ptIn.Color);
    R = cols(:, 1); G = cols(:, 2); B = cols(:, 3);

    exg = 2 * G - R - B;
    gr = (G > R + 2) & (G > B + 2);
    leafMask = exg > max(4, prctile(exg, 58)) | gr;

    pts = ptIn.Location;
    y = pts(:, 2);
    ymid = median(y);

    potMask = ~leafMask & y < (ymid + 0.03);

    if nnz(leafMask) < 20
        leafMask = false(size(leafMask));
    end
    if nnz(potMask) < 20
        potMask = false(size(potMask));
    end

    leafCloud = select(ptIn, leafMask);
    potCloud = select(ptIn, potMask);

    if leafCloud.Count == 0
        leafCloud = pointCloud(zeros(0, 3, 'single'), 'Color', zeros(0, 3, 'uint8'));
    end
    if potCloud.Count == 0
        potCloud = pointCloud(zeros(0, 3, 'single'), 'Color', zeros(0, 3, 'uint8'));
    end
end

function mesh = tryBuildOptionalMesh(ptIn, params)
    mesh = [];

    try
        pt = ptIn;
        if pt.Count > params.mesh_max_points
            stride = max(1, round(pt.Count / params.mesh_max_points));
            idx = 1:stride:pt.Count;
            pt = pointCloud(single(double(pt.Location(idx, :))), 'Color', pt.Color(idx, :));
        end

        pt = pcdownsample(pt, 'gridAverage', 0.0022);
        if pt.Count < 1000
            return;
        end

        normals = pcnormals(pt, 20);
        pt.Normal = orientNormalsToOrigin(normals, double(pt.Location));

        [mesh, ~, density] = pc2surfacemesh(pt, 'poisson', 8);

        if ~isempty(density)
            keep = density > prctile(density, 6);
            if any(~keep)
                mesh = removeLowDensityVertices(mesh, keep);
            end
        end

        mesh = removeDegenerateFaces(mesh);
    catch
        mesh = [];
    end
end

function normals_out = orientNormalsToOrigin(normals_in, points)
    if isempty(normals_in)
        normals_out = normals_in;
        return;
    end

    to_origin = -points;
    d = sum(normals_in .* to_origin, 2);
    normals_out = normals_in;
    flip = d < 0;
    normals_out(flip, :) = -normals_out(flip, :);
end

function mesh_out = removeLowDensityVertices(mesh_in, keep_mask)
    V = mesh_in.Vertices;
    F = mesh_in.Faces;

    keep_idx = find(keep_mask(:));
    map = zeros(size(keep_mask));
    map(keep_idx) = 1:numel(keep_idx);

    face_keep = all(keep_mask(F), 2);
    F2 = map(F(face_keep, :));
    V2 = V(keep_idx, :);

    mesh_out = surfaceMesh(V2, F2);
end

function mesh_out = removeDegenerateFaces(mesh_in)
    V = mesh_in.Vertices;
    F = mesh_in.Faces;

    if isempty(F)
        mesh_out = mesh_in;
        return;
    end

    v1 = V(F(:, 1), :);
    v2 = V(F(:, 2), :);
    v3 = V(F(:, 3), :);

    e1 = v2 - v1;
    e2 = v3 - v1;
    area = 0.5 * sqrt(sum(cross(e1, e2, 2).^2, 2));

    good = isfinite(area) & area > 1e-11;
    mesh_out = surfaceMesh(V, F(good, :));
end

function ok = renderTargetStyle(ptCloud, out_file, params)
    ok = false;

    if isempty(ptCloud) || ptCloud.Count < 100
        return;
    end

    try
        [loc, col] = sampleForRender(ptCloud, params.render_max_points);

        fprintf('  [render] points=%d  col range=[%.3f %.3f]\n', ...
            size(loc,1), min(col(:)), max(col(:)));

        % Visible=on + drawnow so OpenGL context is properly initialised
        f = figure('Visible', 'on', 'Color', 'k', 'Renderer', 'opengl', ...
            'Position', [100, 80, 980, 980]);
        ax = axes('Parent', f);
        styleAxes(ax);
        scatter3(ax, loc(:,1), loc(:,2), loc(:,3), params.render_marker, col, 'filled');
        view(ax, params.render_view(1), params.render_view(2));
        axis(ax, 'equal');
        axis(ax, 'tight');
        grid(ax, 'on');
        box(ax, 'on');
        drawnow;

        exportgraphics(f, out_file, 'Resolution', 250, 'BackgroundColor', 'black');
        close(f);

        if isImageNotBlack(out_file)
            ok = true;
            fprintf('  [render] export OK via exportgraphics+opengl\n');
            return;
        end

        fprintf('  [render] first attempt black, trying print fallback\n');

        % Fallback: print -opengl
        f = figure('Visible', 'on', 'Color', 'k', 'Renderer', 'opengl', ...
            'Position', [100, 80, 980, 980]);
        ax = axes('Parent', f);
        styleAxes(ax);
        scatter3(ax, loc(:,1), loc(:,2), loc(:,3), params.render_marker, col, 'filled');
        view(ax, params.render_view(1), params.render_view(2));
        axis(ax, 'equal');
        axis(ax, 'tight');
        grid(ax, 'on');
        box(ax, 'on');
        drawnow;

        print(f, out_file, '-dpng', '-r250', '-opengl');
        close(f);

        ok = isImageNotBlack(out_file);
        if ok
            fprintf('  [render] export OK via print+opengl\n');
        else
            fprintf('  [render] both attempts produced black image\n');
        end

    catch ME
        warning('renderTargetStyle error: %s\n%s', ME.message, ME.getReport('basic'));
        ok = false;
    end
end


function styleAxes(ax)
    set(ax, 'Color', 'k');
    set(ax, 'XColor', [0.75 0.75 0.75]);
    set(ax, 'YColor', [0.75 0.75 0.75]);
    set(ax, 'ZColor', [0.75 0.75 0.75]);
    set(ax, 'GridColor', [0.42 0.42 0.42]);
    set(ax, 'GridAlpha', 0.42);
    set(ax, 'MinorGridAlpha', 0.2);
    set(ax, 'FontName', 'Consolas');
    set(ax, 'FontSize', 10);
    xlabel(ax, 'X (m)', 'Color', [0.84 0.84 0.84]);
    ylabel(ax, 'Y (m)', 'Color', [0.84 0.84 0.84]);
    zlabel(ax, 'Z (m)', 'Color', [0.84 0.84 0.84]);
end

function [loc, col] = sampleForRender(ptCloud, max_points)
    loc_full = double(ptCloud.Location);
    col_full = double(ptCloud.Color) ./ 255.0;

    n = size(loc_full, 1);
    if n <= max_points
        loc = loc_full;
        col = col_full;
        return;
    end

    idx = round(linspace(1, n, max_points));
    loc = loc_full(idx, :);
    col = col_full(idx, :);
end

function ok = isImageNotBlack(path)
    ok = false;
    if ~exist(path, 'file')
        return;
    end

    I = imread(path);
    if isempty(I)
        return;
    end

    if ndims(I) == 2
        g = double(I) / 255.0;
    else
        g = mean(double(I), 3) / 255.0;
    end

    bright_ratio = mean(g(:) > 0.04);
    avg_intensity = mean(g(:));

    ok = bright_ratio > 0.006 && avg_intensity > 0.012;
end
