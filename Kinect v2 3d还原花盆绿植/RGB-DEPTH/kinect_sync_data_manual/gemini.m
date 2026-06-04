%% ============================================================
% 植株点云重建 FIXED_V3
% 修复内容：
%   1. 手动设置Y_pot范围，关闭错误的自动检测
%   2. 花盆分割增加绿色点过滤（剔除叶片）
%   3. 质心提取改用双轮中值过滤
%   4. 角度计算改用 unwrap() 代替手动跳变修正
%   5. 强制单调时保留小抖动（>5°才修正）
%   6. 优化范围基于实际步进角动态设置
% ============================================================
clear; clc; close all;

%% =========================================================
%  第1步：参数设置
% =========================================================
numFrames   = 25;
fx = 365.456;   fy = 365.456;
cx_i = 254.878; cy_i = 205.395;
depthScale  = 1000;

Z_min = 0.30;  Z_max = 0.85;
Y_roi_min = -0.22;  Y_roi_max = 0.15;

%% ★★★ 核心修复：手动设置花盆Y范围 ★★★
% 根据图3：花盆底峰值≈0.115m，花盆高度约5-8cm
% → 花盆Y范围约 [0.03, 0.12]
% 如果运行后花盆点数不足，尝试扩大为 [0.02, 0.13]
Y_pot_auto = false;
Y_pot      = [0.03, 0.12];

%% 处理参数
voxelSize    = 0.003;
denoiseN     = 20;    denoiseT = 1.0;
densityRadius = 0.005; densityFactor = 3;
outFile      = 'Final_Plant_FIXED_V3.ply';
smoothWeight = 0.3;   maxDist = 0.015;
greenRatio   = 1.25;  greenRatioB = 1.10;  % 绿色判断阈值

fprintf('花盆Y范围: [%.4f, %.4f]\n', Y_pot(1), Y_pot(2));

%% =========================================================
%  第2步：读取点云
% =========================================================
fprintf('\n==== Step 2: 读取点云 ====\n');
ptClouds_raw = cell(numFrames,1);

for i = 1:numFrames
    colorImg = imread(sprintf('%04d_color.tif', i));
    depthImg = imread(sprintf('%04d_depth.tif', i));
    [h,w] = size(depthImg);
    if size(colorImg,1)~=h || size(colorImg,2)~=w
        colorImg = imresize(colorImg,[h,w]);
    end
    [uu,vv] = meshgrid(1:w,1:h);
    Z = double(depthImg)/depthScale;
    X = (uu-cx_i).*Z./fx;
    Y = (vv-cy_i).*Z./fy;
    validDepth = Z>Z_min & Z<Z_max;
    X=X(validDepth); Y=Y(validDepth); Z=Z(validDepth);
    roiY = Y>Y_roi_min & Y<Y_roi_max;
    X=X(roiY); Y=Y(roiY); Z=Z(roiY);
    Rc=colorImg(:,:,1); Gc=colorImg(:,:,2); Bc=colorImg(:,:,3);
    Rc=Rc(validDepth); Gc=Gc(validDepth); Bc=Bc(validDepth);
    Rc=Rc(roiY); Gc=Gc(roiY); Bc=Bc(roiY);
    if numel(X)<100
        ptClouds_raw{i}=pointCloud(zeros(1,3,'single')); continue;
    end
    pc=pointCloud([X,Y,Z],'Color',uint8([Rc,Gc,Bc]));
    pc=pcdownsample(pc,'gridAverage',voxelSize);
    if pc.Count>100
        pc=pcdenoise(pc,'NumNeighbors',denoiseN,'Threshold',denoiseT);
    end
    ptClouds_raw{i}=pc;
    fprintf('Frame%2d: %5d点\n',i,pc.Count);
end

%% =========================================================
%  第3步：Y值分布诊断
% =========================================================
fprintf('\n==== Step 3: Y值分布诊断 ====\n');
allY=[]; allX=[]; allZ=[];
for i=1:numFrames
    if ptClouds_raw{i}.Count>10
        loc=double(ptClouds_raw{i}.Location);
        allY=[allY;loc(:,2)]; allX=[allX;loc(:,1)]; allZ=[allZ;loc(:,3)];
    end
end
fprintf('Y范围: [%.4f, %.4f]\n', min(allY), max(allY));

[counts,edges]=histcounts(allY,150);
centers=(edges(1:end-1)+edges(2:end))/2;

figure('Name','Y值分布诊断','Color','w','Position',[50,50,900,400]);
bar(centers,counts,'FaceColor',[0.3,0.6,0.9],'EdgeColor','none'); hold on;
xline(Y_pot(1),'r--','LineWidth',2.5,'Label',sprintf('花盆顶 %.4f',Y_pot(1)),...
    'LabelVerticalAlignment','bottom');
xline(Y_pot(2),'g--','LineWidth',2.5,'Label',sprintf('花盆底 %.4f',Y_pot(2)),...
    'LabelVerticalAlignment','bottom');
xline(0,'k:','LineWidth',1.5);
xlabel('Y坐标(m)'); ylabel('点数'); grid on;
title(sprintf('★确认红绿虚线之间是否为花盆区域\n花盆Y:[%.4f, %.4f]',Y_pot(1),Y_pot(2)));
nPotPts=sum(allY>=Y_pot(1)&allY<=Y_pot(2));
fprintf('花盆区域点数: %d / %d (%.1f%%)\n',nPotPts,length(allY),100*nPotPts/length(allY));

%% =========================================================
%  第4步：花盆/植株分割 + 颜色过滤
% =========================================================
fprintf('\n==== Step 4: 分割 ====\n');
ptClouds_pot   = cell(numFrames,1);
ptClouds_plant = cell(numFrames,1);

for i=1:numFrames
    if ptClouds_raw{i}.Count<20
        ptClouds_pot{i}=pointCloud(zeros(1,3,'single'));
        ptClouds_plant{i}=pointCloud(zeros(1,3,'single')); continue;
    end
    loc=double(ptClouds_raw{i}.Location);
    col=double(ptClouds_raw{i}.Color);
    maskPot=loc(:,2)>=Y_pot(1) & loc(:,2)<=Y_pot(2);

    %% ★ 颜色过滤：花盆区域剔除绿色点
    nGreenRemoved=0;
    potKeep=maskPot;
    if sum(maskPot)>10
        colPot=col(maskPot,:);
        R_=colPot(:,1); G_=colPot(:,2); B_=colPot(:,3);
        isGreen=(G_>R_*greenRatio)&(G_>B_*greenRatioB);
        potIndices=find(maskPot);
        potKeep(potIndices(isGreen))=false;
        nGreenRemoved=sum(isGreen);
    end

    if sum(potKeep)>=30
        ptClouds_pot{i}=pointCloud(single(loc(potKeep,:)),'Color',uint8(col(potKeep,:)));
    else
        ptClouds_pot{i}=pointCloud(zeros(1,3,'single'));
    end

    plantMask=~maskPot;
    if sum(maskPot)>10
        potIndicesAll=find(maskPot);
        colPot2=col(maskPot,:);
        isGreen2=(colPot2(:,2)>colPot2(:,1)*greenRatio)&...
                 (colPot2(:,2)>colPot2(:,3)*greenRatioB);
        greenInPot=false(size(loc,1),1);
        greenInPot(potIndicesAll(isGreen2))=true;
        plantMask=plantMask|greenInPot;
    end
    ptClouds_plant{i}=pointCloud(single(loc(plantMask,:)),'Color',uint8(col(plantMask,:)));
    fprintf('Frame%2d: 花盆=%4d  植株=%5d  (绿色剔除=%d)\n',...
        i,ptClouds_pot{i}.Count,ptClouds_plant{i}.Count,nGreenRemoved);
end

%% =========================================================
%  第5步：鲁棒质心提取
% =========================================================
fprintf('\n==== Step 5: 质心提取 ====\n');
centroids=NaN(numFrames,3);
centroid_std=NaN(numFrames,1);

for i=1:numFrames
    if ptClouds_pot{i}.Count<20, continue; end
    pts=double(ptClouds_pot{i}.Location);
    % 第1轮过滤
    med=median(pts(:,[1,3]),1);
    d=sqrt((pts(:,1)-med(1)).^2+(pts(:,3)-med(2)).^2);
    pts1=pts(d<=max(quantile(d,0.70),0.003),:);
    % 第2轮过滤
    if size(pts1,1)>=10
        med2=median(pts1(:,[1,3]),1);
        d2=sqrt((pts1(:,1)-med2(1)).^2+(pts1(:,3)-med2(2)).^2);
        pts2=pts1(d2<=max(quantile(d2,0.60),0.002),:);
    else
        pts2=pts1;
    end
    if size(pts2,1)<5, continue; end
    centroids(i,:)=mean(pts2,1);
    centroid_std(i)=mean(std(pts2(:,[1,3])));
end

validIdx=find(~any(isnan(centroids),2));
fprintf('有效帧: %d/%d\n',length(validIdx),numFrames);

xRange=range(centroids(validIdx,1));
zRange=range(centroids(validIdx,3));
fprintf('质心XZ范围: X=%.5fm Z=%.5fm\n',xRange,zRange);
if xRange<0.005 && zRange<0.005
    error('★ 质心范围过小(%.2fmm)！Y_pot范围不正确，请重新检查Step3图',...
        max(xRange,zRange)*1000);
end

%% =========================================================
%  第6步：剔除不可信帧 + RANSAC圆拟合
% =========================================================
fprintf('\n==== Step 6: 圆拟合 ====\n');
std_thresh=median(centroid_std(validIdx),'omitnan')*2.5;
reliable=validIdx(centroid_std(validIdx)<=std_thresh);
fprintf('可信帧: %d/%d\n',length(reliable),length(validIdx));

goodIdx=reliable;
cx0=0; cz0=0; r0=0;
for iter=1:10
    if length(goodIdx)<4, break; end
    xv=centroids(goodIdx,1); zv=centroids(goodIdx,3);
    [cx0,cz0,r0]=fitCircleRANSAC(xv,zv,0.008);
    dists=abs(sqrt((xv-cx0).^2+(zv-cz0).^2)-r0);
    keepMask=dists<=0.010;
    newGood=goodIdx(keepMask);
    fprintf('  iter%d: 圆心(%.5f,%.5f) r=%.5fm 剔除%d帧\n',...
        iter,cx0,cz0,r0,sum(~keepMask));
    if length(newGood)==length(goodIdx)||length(newGood)<5, break; end
    goodIdx=newGood;
end
fprintf('好帧(%d): ',length(goodIdx)); fprintf('%d ',goodIdx); fprintf('\n');
fprintf('圆心=(%.5f,%.5f) r=%.5fm (%.1fmm)\n',cx0,cz0,r0,r0*1000);
if r0<0.005
    fprintf('[警告] 圆半径过小！花盆分割仍有问题\n');
end

%% =========================================================
%  第7步：旋转角度（unwrap法）
% =========================================================
fprintf('\n==== Step 7: 旋转角度 ====\n');
rotY=@(theta)[cos(theta),0,sin(theta);0,1,0;-sin(theta),0,cos(theta)];
center0=[cx0;0;cz0];

rawAng=zeros(length(goodIdx),1);
for k=1:length(goodIdx)
    i=goodIdx(k);
    rawAng(k)=atan2(centroids(i,3)-cz0,centroids(i,1)-cx0);
end

%% unwrap + 归零 + 确保递增
unwrappedAng=unwrap(rawAng);
unwrappedAng=unwrappedAng-unwrappedAng(1);
if unwrappedAng(end)<0, unwrappedAng=-unwrappedAng; end

for k=2:length(unwrappedAng)
    if unwrappedAng(k)<unwrappedAng(k-1)-deg2rad(5)
        fprintf('[修正] Frame%2d->%2d 回退%.1f°→拉平\n',...
            goodIdx(k-1),goodIdx(k),...
            rad2deg(unwrappedAng(k-1)-unwrappedAng(k)));
        unwrappedAng(k)=unwrappedAng(k-1);
    end
end
cumAng=unwrappedAng;

fprintf('各帧角度:\n');
for k=1:length(goodIdx)
    fprintf('  Frame%2d: %.2f deg\n',goodIdx(k),rad2deg(cumAng(k)));
end
fprintf('总转角: %.1f deg\n',rad2deg(cumAng(end)));

%% 插值全帧
angles_all=NaN(numFrames,1);
angles_all(goodIdx)=cumAng;
knownF=find(~isnan(angles_all));
angles_all=interp1(knownF,angles_all(knownF),1:numFrames,'linear','extrap')';
angles_all=max(angles_all,0);

%% 诊断图
figure('Name','FIXED_V3诊断','Color','w','Position',[50,50,1400,500]);
subplot(1,3,1);
scatter(centroids(goodIdx,1),centroids(goodIdx,3),80,'g','filled'); hold on;
th=linspace(0,2*pi,200);
plot(cx0+r0*cos(th),cz0+r0*sin(th),'b-','LineWidth',2);
plot(cx0,cz0,'b+','MarkerSize',15,'LineWidth',3);
for k=1:length(goodIdx)
    text(centroids(goodIdx(k),1)+r0*0.03,centroids(goodIdx(k),3),...
        num2str(goodIdx(k)),'FontSize',8);
end
axis equal; grid on; xlabel('X(m)'); ylabel('Z(m)');
title(sprintf('质心分布 好帧%d个 r=%.4fm',length(goodIdx),r0));

subplot(1,3,2);
plot(goodIdx,rad2deg(cumAng),'go-','LineWidth',2,'MarkerSize',8); hold on;
plot(1:numFrames,rad2deg(angles_all),'b--','LineWidth',1.5);
xlabel('帧'); ylabel('度'); grid on;
legend('实测','插值');
title(sprintf('旋转角度曲线\n总转角=%.1f°（应为S形或直线）',rad2deg(cumAng(end))));

subplot(1,3,3);
cmap=lines(numFrames);
for i=1:numFrames
    if ptClouds_pot{i}.Count<10, continue; end
    pts=double(ptClouds_pot{i}.Location);
    R=rotY(-angles_all(i)); t=center0-R*center0;
    pts_t=(R*pts'+t)';
    scatter(pts_t(:,1),pts_t(:,3),3,cmap(i,:),'filled'); hold on;
end
plot(cx0,cz0,'k+','MarkerSize',15,'LineWidth',3);
axis equal; grid on; xlabel('X'); ylabel('Z');
title('花盆XZ对齐（颜色混合越好=对齐越准）');

%% =========================================================
%  第8步：全局优化
% =========================================================
fprintf('\n==== Step 8: 全局优化 ====\n');
ptClouds_opt=cell(numFrames,1);
for i=1:numFrames
    loc=[]; col=[];
    if ptClouds_pot{i}.Count>10
        loc=[loc;double(ptClouds_pot{i}.Location)];
        col=[col;double(ptClouds_pot{i}.Color)];
    end
    if ptClouds_plant{i}.Count>10
        loc=[loc;double(ptClouds_plant{i}.Location)];
        col=[col;double(ptClouds_plant{i}.Color)];
    end
    if ~isempty(loc)
        ptClouds_opt{i}=pcdownsample(...
            pointCloud(single(loc),'Color',uint8(col)),'gridAverage',0.006);
    else
        ptClouds_opt{i}=pointCloud(zeros(1,3,'single'));
    end
end

medStep=median(abs(diff(angles_all)));
angTol=max(medStep*0.6, deg2rad(3));

x0=[cx0;cz0;angles_all(2:end)];
lb=[cx0-0.020;cz0-0.020;angles_all(2:end)-angTol];
ub=[cx0+0.020;cz0+0.020;angles_all(2:end)+angTol];

errFn=@(x)globalAlignError(x,ptClouds_opt,rotY,maxDist,smoothWeight);
opts=optimoptions('lsqnonlin','Display','iter',...
    'MaxFunctionEvaluations',8000,'MaxIterations',200,...
    'FunctionTolerance',1e-7,'StepTolerance',1e-7);
try
    [x_opt,resnorm]=lsqnonlin(errFn,x0,lb,ub,opts);
    cx_opt=x_opt(1); cz_opt=x_opt(2);
    angles_opt=[0;x_opt(3:end)];
    fprintf('优化完成 残差=%.5f\n',resnorm);
catch ME
    fprintf('优化失败(%s)，使用初始角度\n',ME.message);
    cx_opt=cx0; cz_opt=cz0; angles_opt=angles_all;
end
center_opt=[cx_opt;0;cz_opt];

%% =========================================================
%  第9步：融合、去噪、保存
% =========================================================
fprintf('\n==== Step 9: 融合 ====\n');
allXYZ=[]; allRGB=[];
for i=1:numFrames
    loc=[]; col=[];
    if ptClouds_pot{i}.Count>10
        loc=[loc;double(ptClouds_pot{i}.Location)];
        col=[col;double(ptClouds_pot{i}.Color)];
    end
    if ptClouds_plant{i}.Count>10
        loc=[loc;double(ptClouds_plant{i}.Location)];
        col=[col;double(ptClouds_plant{i}.Color)];
    end
    if isempty(loc), continue; end
    R=rotY(-angles_opt(i)); t=center_opt-R*center_opt;
    pts_t=(R*loc'+t)';
    allXYZ=[allXYZ;pts_t]; allRGB=[allRGB;col];
end

voxIdx=floor(allXYZ./voxelSize);
[~,ia]=unique(voxIdx,'rows','stable');
mergedCloud=pointCloud(single(allXYZ(ia,:)),'Color',uint8(allRGB(ia,:)));
fprintf('降采样后: %d点\n',mergedCloud.Count);

mergedCloud=removeHighDensityPoints_serial(mergedCloud,densityRadius,densityFactor);
if mergedCloud.Count>100
    mergedCloud=pcdenoise(mergedCloud,'NumNeighbors',25,'Threshold',0.8);
end

normals=pcnormals(mergedCloud,20);
mergedCloud.Normal=normals;
pcwrite(mergedCloud,outFile,'Encoding','binary');
fprintf('已保存: %s\n',outFile);

figure('Name','最终点云 FIXED_V3','Color','k','Position',[100,100,1000,800]);
pcshow(mergedCloud,'VerticalAxis','Y','VerticalAxisDir','Down','MarkerSize',6);
view(-35,25); grid on;
title(sprintf('最终结果  总转角=%.1f°  %d点',...
    rad2deg(max(angles_opt)),mergedCloud.Count),'Color','w');
set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');
disp('==== 完成 ====');

%% =========================================================
%  局部函数
% =========================================================
function [cx,cz,radius]=fitCircleRANSAC(x,z,threshold)
    x=x(:);z=z(:);n=length(x);
    if n<3,cx=mean(x);cz=mean(z);radius=0.05;return;end
    if n<5,[cx,cz,radius]=fitCircleLS(x,z);return;end
    bestCount=0;bestInliers=true(n,1);
    for iter=1:800
        idx=randperm(n,3);
        x1=x(idx(1));z1=z(idx(1));x2=x(idx(2));z2=z(idx(2));
        x3=x(idx(3));z3=z(idx(3));
        A=[x1,z1,1;x2,z2,1;x3,z3,1];
        if abs(det(A))<1e-10,continue;end
        B=-[x1^2+z1^2;x2^2+z2^2;x3^2+z3^2];
        p=A\B;cx_t=-p(1)/2;cz_t=-p(2)/2;
        r2=p(1)^2/4+p(2)^2/4-p(3);
        if r2<=0,continue;end
        d=abs(sqrt((x-cx_t).^2+(z-cz_t).^2)-sqrt(r2));
        inl=d<threshold;
        if sum(inl)>bestCount,bestCount=sum(inl);bestInliers=inl;end
    end
    if sum(bestInliers)>=3,[cx,cz,radius]=fitCircleLS(x(bestInliers),z(bestInliers));
    else,[cx,cz,radius]=fitCircleLS(x,z);end
end

function [cx,cz,radius]=fitCircleLS(x,z)
    x=x(:);z=z(:);
    A=[x,z,ones(size(x))];B=-(x.^2+z.^2);
    p=A\B;cx=-p(1)/2;cz=-p(2)/2;
    radius=sqrt(max(0,p(1)^2/4+p(2)^2/4-p(3)));
end

function residuals=globalAlignError(params,ptClouds,rotY,maxDist,smoothWeight)
    params=params(:);cx=params(1);cz=params(2);
    center=[cx;0;cz];angles=[0;params(3:end)];
    N=length(ptClouds);maxPts=300;
    refIdx=1;
    for k=1:N,if ptClouds{k}.Count>=20,refIdx=k;break;end,end
    refPts=double(ptClouds{refIdx}.Location);
    if size(refPts,1)>maxPts,refPts=refPts(randperm(size(refPts,1),maxPts),:);end
    R0=rotY(-angles(refIdx));t0=center-R0*center;
    refPts_t=(R0*refPts'+t0)';
    refTree=KDTreeSearcher(refPts_t);
    distRes=[];
    for i=1:N
        if i==refIdx||ptClouds{i}.Count<10,continue;end
        pts=double(ptClouds{i}.Location);
        if size(pts,1)>maxPts,pts=pts(randperm(size(pts,1),maxPts),:);end
        R=rotY(-angles(i));t=center-R*center;
        pts_t=(R*pts'+t)';
        [~,d]=knnsearch(refTree,pts_t,'K',1);
        distRes=[distRes;min(d,maxDist)];
    end
    smoothRes=zeros(N-2,1);
    for i=2:N-1
        smoothRes(i-1)=sqrt(smoothWeight)*(angles(i+1)-2*angles(i)+angles(i-1));
    end
    monoRes=zeros(N-1,1);
    for i=1:N-1
        d=angles(i+1)-angles(i);
        if d<0,monoRes(i)=sqrt(smoothWeight)*3*abs(d);end
    end
    if isempty(distRes),residuals=1e6*ones(50,1);
    else,residuals=[distRes;smoothRes;monoRes];end
end

function ptCloudOut=removeHighDensityPoints_serial(ptCloud,radius,densityFactor)
    loc=ptCloud.Location;
    if isempty(loc)||size(loc,1)<100,ptCloudOut=ptCloud;return;end
    tree=KDTreeSearcher(loc);
    densities=zeros(size(loc,1),1);
    for i=1:size(loc,1)
        idx=rangesearch(tree,loc(i,:),radius);
        densities(i)=length(idx{1});
    end
    thresh=mean(densities)*densityFactor;
    keep=densities<=thresh;
    ptCloudOut=select(ptCloud,keep);
    fprintf('密度去重: 移除%d点(%.1f%%)\n',sum(~keep),100*mean(double(~keep)));
end