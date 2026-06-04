% Visualization-only script with software OpenGL to avoid GPU crash
opengl software;
fprintf('Using software OpenGL rendering\n');

OUTPUT_DIR = 'E:/RGB-DEPTH/results';

% Load results
ptCloud_raw = pcread(fullfile(OUTPUT_DIR, 'cloud_raw.ply'));
ptCloud_clean = pcread(fullfile(OUTPUT_DIR, 'cloud_clean.ply'));
ptCloud_leaf = pcread(fullfile(OUTPUT_DIR, 'cloud_leaf.ply'));
ptCloud_pot = pcread(fullfile(OUTPUT_DIR, 'cloud_pot.ply'));
ptCloud_plant = pcread(fullfile(OUTPUT_DIR, 'cloud_plant.ply'));
mesh = readSurfaceMesh(fullfile(OUTPUT_DIR, 'mesh_plant.ply'));

fprintf('Loaded: raw=%d, clean=%d, leaf=%d, pot=%d, plant=%d\n', ...
    ptCloud_raw.Count, ptCloud_clean.Count, ptCloud_leaf.Count, ...
    ptCloud_pot.Count, ptCloud_plant.Count);
fprintf('Mesh: %d vertices, %d faces\n', mesh.NumVertices, mesh.NumFaces);

%% Figure 1: Pipeline overview
fprintf('Generating pipeline overview...\n');
fig1 = figure('Position', [50, 50, 1800, 900], 'Visible', 'off');

subplot(2, 3, 1);
pcshow(ptCloud_raw, 'MarkerSize', 1); title('Raw Point Cloud');

subplot(2, 3, 2);
pcshow(ptCloud_clean, 'MarkerSize', 2); title('Cleaned + ROI Filtered');

subplot(2, 3, 3);
pcshow(ptCloud_leaf, 'MarkerSize', 3); title(sprintf('Leaf (%d pts)', ptCloud_leaf.Count));

subplot(2, 3, 4);
if ptCloud_pot.Count > 0
    pcshow(ptCloud_pot, 'MarkerSize', 3);
end
title(sprintf('Pot (%d pts)', ptCloud_pot.Count));

subplot(2, 3, 5);
pcshow(ptCloud_plant, 'MarkerSize', 3);
title(sprintf('Plant+Pot (%d pts)', ptCloud_plant.Count));

subplot(2, 3, 6);
if mesh.NumFaces > 0
    trisurf(mesh.Faces, mesh.Vertices(:,1), mesh.Vertices(:,2), mesh.Vertices(:,3), ...
        'FaceColor', [0.25, 0.6, 0.25], 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    camlight('headlight'); lighting gouraud;
end
title('Surface Mesh');
axis equal; drawnow;

saveas(fig1, fullfile(OUTPUT_DIR, 'pipeline_overview.png'));
close(fig1);
fprintf('  Saved: pipeline_overview.png\n');

%% Figure 2: Multi-angle mesh views
fprintf('Generating mesh views...\n');
fig2 = figure('Position', [50, 50, 1600, 1200], 'Visible', 'off');
views = {[0, 90], [90, 0], [45, 30], [-45, 25]};
names = {'Top View', 'Front View', 'Perspective 1', 'Perspective 2'};
for i = 1:4
    subplot(2, 2, i);
    if mesh.NumFaces > 0
        trisurf(mesh.Faces, mesh.Vertices(:,1), mesh.Vertices(:,2), mesh.Vertices(:,3), ...
            'FaceColor', [0.25, 0.55, 0.25], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
        camlight('headlight'); lighting gouraud; material dull;
    else
        pcshow(ptCloud_plant, 'MarkerSize', 4);
    end
    view(views{i}); axis equal;
    title(names{i});
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
end
drawnow;
saveas(fig2, fullfile(OUTPUT_DIR, 'mesh_views.png'));
close(fig2);
fprintf('  Saved: mesh_views.png\n');

%% Figure 3: Plant point cloud detail
fprintf('Generating plant point cloud...\n');
fig3 = figure('Position', [50, 50, 1400, 900], 'Visible', 'off');
pcshow(ptCloud_plant, 'MarkerSize', 5);
title('Plant + Pot - Colored Point Cloud');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view([-45, 30]); camlight; drawnow;
saveas(fig3, fullfile(OUTPUT_DIR, 'plant_pointcloud.png'));
close(fig3);
fprintf('  Saved: plant_pointcloud.png\n');

fprintf('\nVisualization complete!\n');
