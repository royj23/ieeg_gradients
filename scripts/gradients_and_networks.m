%% init

for init_project = 1
    
    % directories
    GH  = '/path/to/your/repo/'; 
    project_repo = [GH, 'ieeg_gradients/']; % github directory
    datadir = [project_repo, 'data/'];
    outdir = [project_repo, 'outputs/'];
    utildir = [project_repo, 'utilities/'];

    % add dependencies to path
    addpath(genpath(project_repo));
    addpath(genpath([GH, '/BrainSpace/matlab'])); %Brainspace: for gradients
    addpath(genpath([GH, 'plotSurfaceROIBoundary'])); % for plotting on cortical surface
    addpath(genpath([GH, 'gifti-1.6/'])); % loading surfaces
    
    % colormaps   
    load([utildir, 'roma.mat']); % credit to scientific colormaps (https://www.fabiocrameri.ch/colourmaps/)
    load([utildir, 'davos.mat']); % credit to scientific colormaps (https://www.fabiocrameri.ch/colourmaps/)
    load([utildir, 'mri_colours.mat']) % credit to colorbrewer
    
    % Cortical surface data
    % Left and right surfaces, recode values in SurfStat naming
    tmp = gifti([utildir, 'fsLR-32k.L.inflated.surf.gii']);
    c69L = struct(); c69L.vertices = tmp.vertices; c69L.faces = tmp.faces;
    tmp = gifti([utildir, 'fsLR-32k.R.inflated.surf.gii']);
    c69R = struct(); c69R.vertices = tmp.vertices; c69R.faces = tmp.faces;
    clear tmp
    % Combine hemispheres
    c69 = struct();
    c69.coord = [c69L.vertices', c69R.vertices'];
    c69.tri = [c69L.faces; c69R.faces+length(c69R.faces)];
    % Medial wall mask
    maskL = gifti([utildir, 'conte69_32k_lh_mask.surf.gii']); maskL = maskL.cdata;
    maskR = gifti([utildir, 'conte69_32k_rh_mask.surf.gii']); maskR = maskR.cdata;
    mask = logical([maskL;maskR])';
    % number of vertices in the surface
    nFS = length(c69L.vertices) + length(c69R.vertices);

    % Axis parameters for plotting
    plot_brains = [0.2 0.3 0.28 0.28; 0.49 0.3 0.28 0.28];

    % Load channel location on conte69 surface
    load([utildir, 'vertexPatchMatching_C69.mat']);
    
end


%% Load parcellation

parcName='schaefer-200';

for load_parc = 1
    
    % Read parccellation for C69
    parcFS = csvread([utildir, parcName, '_conte69.csv']);
    
    uparc = unique(parcFS);
    parcFSR = parcFS(nFS/2+1:end);
    uparcR = unique(parcFSR);
    
    maskCol = 0; % value of mask in parcellation
    maskIdxFS = find(parcFS == maskCol(1));
    maskParc = zeros(1,nFS); maskParc(1,maskIdxFS) = 1;
    
end


%% Load other dependencies

% PSDs gradients
load([datadir, 'gradients-schaefer-200.mat']);

% MRI modalities: Average matrices across 50 MICA-MICs subjects
SCMR = csvread([datadir, 'SC.csv']);
MPCMR = csvread([datadir, 'MPC.csv']);
FCMR = csvread([datadir, 'FC.csv']);
GDMR = csvread([datadir, 'GD.csv']);
rTrue_all = zeros(4,1);

% Load permuted parcellation for null models
nrot = 1000;
load([utildir, 'permutations_Schaefer200.mat']);
perm_idR = perm_id(101:end,:)-100;


%% Compute embedding distance matrix

% euclidean distance matrix
DE = [];
G = G(:,1:nG_keep);
for ii = 1:size(G,1)
    for jj = 1:size(G,1)
        DE(ii,jj) = sqrt(sum((G(ii,:)-G(jj,:)).^2));
    end
end
distRange = [0 0.3];

fig = figure,
    imagesc(DE); colormap(davos)
    axis('square'); caxis(distRange);
    xticklabels(''); yticklabels(''); xticks(''); yticks(''); 


%% Run correlations: GD

% Get empirical correlation
tmp = inf(size(GDMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask to only keep upper triangle
tmp2 = tmp .* GDMR; % apply mask
tmp2 = tmp2(:);
keep = ~isnan(tmp2) & ~isinf(tmp2) & tmp2~=0; % remove edges with inf, nan, or 0
GD_tri = tmp2(keep); 

dist_tri = triu(DE,1);
dist_tri = dist_tri(:);
dist_tri = dist_tri(keep); % remove the same excluded edges from ED matrix

rTrue = corr(dist_tri,GD_tri);
rTrue_all(1) = rTrue;

% Figure : scatter plot
for figure_time = 1
    fig = figure,
    a(1) = axes('position', [0.02    0.2000    0.5200    0.5200]);
    scatter(GD_tri, dist_tri, 30, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerEdgeAlpha', 0); hold on
    h1 = lsline(a(1));
    h1.LineWidth = 1.5;
    h1.Color = [1 0 0];
    xlabel('Geodesic distance (inverse)')
    ylabel('Embedding distance')
    ylim([0 0.35]); yticks([0 0.1 0.2 0.3])
    xlim([-550 0]); xticks([-550:100:0])
    axis('square');
end

% GD null: Reorder matrix
GDMR_null = [];
for ii = 1:size(GDMR,1)
    for jj = 1:nrot
        this_perm = perm_idR(:,jj);
        GDMR_null(ii,:,jj) = GDMR(ii,this_perm);
    end
end

% test if rTrue is significant against null model
rPerm_GD = [];
for ii = 1:nrot
    thisPerm = GDMR_null(:,:,ii);
    
    tmp = inf(size(thisPerm)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmp2 = tmp .* thisPerm;
    tmp2 = tmp2(:);
    
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    
    keep = ~isnan(tmp2) & ~isinf(tmp2) & dist_tri~=0;
    
    GD_tri_perm = tmp2(keep);
    dist_tri = dist_tri(keep);
    
    rPerm_GD(ii) = corr(dist_tri,GD_tri_perm);
end
pvalGD_weights = sum(abs(rPerm_GD) > abs(rTrue))/nrot;


%% Run correlations: SC

% Get SC weights
tmp = inf(size(SCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
tmp2 = tmp .* SCMR;
tmp2 = tmp2(:);
keep = ~isnan(tmp2) & ~isinf(tmp2) & tmp2~=0;
SC_tri = tmp2(keep);

dist_tri = triu(DE,1);
dist_tri = dist_tri(:);
dist_tri = dist_tri(keep);

rTrue = corr(dist_tri,SC_tri);
rTrue_all(2) = rTrue;

% Figure: scatter plot
for figure_time = 1
    
    fig = figure,
    a(1) = axes('position', [0.05    0.2000    0.5200    0.5200]);
    scatter(SC_tri, dist_tri, 30, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerEdgeAlpha', 0); hold on
    h1 = lsline(a(1));
    h1.LineWidth = 1.5;
    h1.Color = [1 0 0];
    xlabel('SC Weight')
    ylabel('Embedding distance')
    ylim([0 0.35]); yticks([0 0.1 0.2 0.3])
    xlim([0 12]); xticks([0 6 12])
    axis('square');
    
end

% SC null: Reorder matrix
SCMR_null = [];
for ii = 1:size(SCMR,1)
    for jj = 1:nrot
        this_perm = perm_idR(:,jj);
        SCMR_null(ii,:,jj) = SCMR(ii,this_perm);
    end
end

% test if rTrue is significant against null model
rPerm_SC = [];
for ii = 1:nrot
    thisPerm = SCMR_null(:,:,ii);
    
    tmp = inf(size(thisPerm)); tmp = tril(thisPerm); tmp(tmp==0) = 1; % Create mask
    tmp2 = tmp .* thisPerm;
    tmp2 = tmp2(:);
    
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    
    keep = ~isnan(tmp2) & ~isinf(tmp2) & tmp2~=0 & dist_tri~=0;
    
    SC_tri_perm = tmp2(keep);
    dist_tri = dist_tri(keep);
    
    rPerm_SC(ii) = corr(dist_tri,SC_tri_perm);
end
pvalSC_weights = sum(abs(rPerm_SC) > abs(rTrue))/nrot;


%% Run correlations: FC

% Get FC weights
tmp = inf(size(FCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
tmp2 = tmp .* FCMR;
tmp2 = tmp2(:);
keep = ~isnan(tmp2) & ~isinf(tmp2) & tmp2>0;
FC_tri = tmp2(keep);

dist_tri = triu(DE,1);
dist_tri = dist_tri(:);
dist_tri = dist_tri(keep);

rTrue = corr(dist_tri,FC_tri);
rTrue_all(3) = rTrue;

% Figure: scatter plot
for figure_time = 1
    fig = figure,
    a(1) = axes('position', [0.02    0.2000    0.5200    0.5200]);
    scatter(FC_tri, dist_tri, 30, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerEdgeAlpha', 0); hold on
    h1 = lsline(a(1));
    h1.LineWidth = 1.5;
    h1.Color = [1 0 0];
    xlabel('FC weight')
    ylabel('Embedding distance')
    ylim([0 0.35]); yticks([0 0.1 0.2 0.3])
    xlim([0 1.4]); xticks([0:0.3:1.4])
    axis('square');
end

% FC null: Reorder matrix
FCMR_null = [];
for ii = 1:size(FCMR,1)
    for jj = 1:nrot
        this_perm = perm_idR(:,jj);
        FCMR_null(ii,:,jj) = FCMR(ii,this_perm);
    end
end

% test if rTrue is significant against null model
rPerm_FC = [];
for ii = 1:nrot
    thisPerm = FCMR_null(:,:,ii);
    
    tmp = inf(size(thisPerm)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmp2 = tmp .* thisPerm;
    tmp2 = tmp2(:);
    
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    
    keep = ~isnan(tmp2) & ~isinf(tmp2) & dist_tri>0;
    
    FC_tri_perm = tmp2(keep);
    dist_tri = dist_tri(keep);
    
    rPerm_FC(ii) = corr(dist_tri,FC_tri_perm);
end
pvalFC_weights = sum(abs(rPerm_FC) > abs(rTrue))/nrot;


%% Run correlations: MPC

% Get MPC weights
tmp = inf(size(MPCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
tmp2 = tmp .* MPCMR;
tmp2 = tmp2(:);
keep = ~isnan(tmp2) & ~isinf(tmp2);
MPC_tri = tmp2(keep);

dist_tri = triu(DE,1);
dist_tri = dist_tri(:);
dist_tri = dist_tri(keep);

rTrue = corr(dist_tri,MPC_tri);
rTrue_all(4) = rTrue;

% Figure: scatter plot
for figure_time = 1
    fig = figure,
    a(1) = axes('position', [0.02    0.2000    0.5200    0.5200]);
    scatter(MPC_tri, dist_tri, 30, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.1, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerEdgeAlpha', 0); hold on
    h1 = lsline(a(1));
    h1.LineWidth = 1.5;
    h1.Color = [1 0 0];
    xlabel('Edge Weight')
    ylabel('Embedding distance')
    ylim([0 0.35]); yticks([0 0.1 0.2 0.3])
    xlim([-3 4]); xticks([-2 0 2 4])
    axis('square');
end

% MPC null: Reorder matrix
MPCMR_null = [];
for ii = 1:size(MPCMR,1)
    for jj = 1:nrot
        this_perm = perm_idR(:,jj);
        MPCMR_null(ii,:,jj) = MPCMR(ii,this_perm);
    end
end

% test if rTrue is significant against null model
rPerm_MPC = [];
for ii = 1:nrot
    thisPerm = MPCMR_null(:,:,ii);
    
    tmp = inf(size(thisPerm)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmp2 = tmp .* thisPerm;
    tmp2 = tmp2(:);
    
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    
    keep = ~isnan(tmp2) & ~isinf(tmp2) & dist_tri~=0;
    
    MPC_tri_perm = tmp2(keep);
    dist_tri = dist_tri(keep);
    
    rPerm_MPC(ii) = corr(dist_tri,MPC_tri_perm);
end
pvalMPC_weights = sum(abs(rPerm_MPC) > abs(rTrue))/nrot;

