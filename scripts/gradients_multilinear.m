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
    addpath(genpath([GH, '/BrainSpace/matlab'])); % Brainspace: for gradients
    addpath(genpath([GH, 'plotSurfaceROIBoundary'])); % for plotting on cortical surface
    addpath(genpath([GH, 'gifti-1.6/'])); % loading surfaces
    
    % colormaps   
    load([utildir, 'roma.mat']); % credit to scientific colormaps (https://www.fabiocrameri.ch/colourmaps/)
    load([utildir, 'davos.mat']); % credit to scientific colormaps (https://www.fabiocrameri.ch/colourmaps/)
    load([utildir, 'vik.mat']); % credit to scientific colormaps (https://www.fabiocrameri.ch/colourmaps/)
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
DE = csvread([datadir, 'DE.csv']);

% Load permuted parcellation for null models
nrot = 1000;
load([utildir, 'permutations_Schaefer200.mat']);
perm_idR = perm_id(101:end,:)-100;


%% Multilinear model over all edges

% Baseline, partial and full models: DE ~ b0 + b1*GD + (b2*SC + b3*FC + b4*MPC)
% Note: edges are filtered across all modalities (common edges for all)
for baseline_partial_full_models = 1
    
    % Add each feature and combinations
    tmp = inf(size(GDMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpGD = tmp .* GDMR;
    tmpGD = tmpGD(:);
    
    tmp = inf(size(FCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpFC = tmp .* FCMR;
    tmpFC = tmpFC(:);
    
    tmp = inf(size(MPCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpMPC = tmp .* MPCMR;
    tmpMPC = tmpMPC(:);
    
    tmp = inf(size(SCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpSC = tmp .* SCMR;
    tmpSC = tmpSC(:);
    
    keep = ~isnan(tmpGD) & ~isinf(tmpGD) & tmpGD~=0 & ...
        ~isnan(tmpMPC) & ~isinf(tmpMPC) & ... 
        ~isnan(tmpFC) & ~isinf(tmpFC) & tmpFC>0 & ...
        ~isnan(tmpSC) & ~isinf(tmpSC) & tmpSC~=0;
    GD_tri = tmpGD(keep);
    SC_tri = tmpSC(keep);
    FC_tri = tmpFC(keep);
    MPC_tri = tmpMPC(keep);
    
    % Predictors
    X = {};
    X{1} = GD_tri;
    X{2} = [GD_tri, SC_tri];
    X{3} = [GD_tri, FC_tri];
    X{4} = [GD_tri, MPC_tri];
    X{5} = [GD_tri, SC_tri, FC_tri];
    X{6} = [GD_tri, SC_tri, MPC_tri];
    X{7} = [GD_tri, FC_tri, MPC_tri];
    X{8} = [GD_tri, SC_tri, FC_tri, MPC_tri];
    
    % Response
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    dist_tri = dist_tri(keep);
    y = dist_tri;

    % fit regression
    DE_pred_partial = {};
    DE_R2_partial = zeros(size(X,2),1);
    DE_T_partial = {};
    DE_pred_MSE = [];
    ypred_matrix = zeros(size(DE,1), size(DE,2), size(X,2));
    ypred_matrix_diff = zeros(size(DE,1), size(DE,2), size(X,2));
    
    for ii = 1:size(X,2)
        thisX = X{ii};
        mdl = fitlm(thisX,y);
        % uncomment next line to save all model parameters/fit
        %save([outdir, 'all_edges-mdl_',char(string(ii)),'.mat'],'mdl');
        
        ypred = predict(mdl,thisX)';
        DE_pred_partial{ii} = ypred;
        DE_R2_partial(ii) = mdl.Rsquared.Adjusted;
        DE_T_partial{ii} = mdl.Coefficients.tStat';
        DE_pred_MSE(ii) = mean((y - ypred') .^2);
        
        ypred_tmp = zeros(1,size(DE,1)*size(DE,2));
        ypred_tmp(keep) = ypred;
        ypred_matrix(:,:,ii) = reshape(ypred_tmp, size(DE));
        
        ypred_diff_tmp = zeros(1,size(DE,1)*size(DE,2));
        ypred_diff_tmp(keep) = (y - ypred') .^2;
        ypred_matrix_diff(:,:,ii) = reshape(ypred_diff_tmp, size(DE));
    end
    
end


%% Multilinear model over all edges: bootstrapping
% Estimate parameters from 90% data and estimate y on 10%
% * NOTE this take a while to run!

nrep = 1000;
split = 0.9;
rng('shuffle');

for bootstrapping = 1
    
    % Add each feature and combinations
    tmp = inf(size(GDMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpGD = tmp .* GDMR;
    tmpGD = tmpGD(:);
    
    tmp = inf(size(FCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpFC = tmp .* FCMR;
    tmpFC = tmpFC(:);
    
    tmp = inf(size(MPCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpMPC = tmp .* MPCMR;
    tmpMPC = tmpMPC(:);
    
    tmp = inf(size(SCMR)); tmp = tril(tmp); tmp(tmp==0) = 1; % Create mask
    tmpSC = tmp .* SCMR;
    tmpSC = tmpSC(:);
    
    keep = ~isnan(tmpGD) & ~isinf(tmpGD) & tmpGD~=0 & ...
        ~isnan(tmpMPC) & ~isinf(tmpMPC) & ...
        ~isnan(tmpFC) & ~isinf(tmpFC) & tmpFC>0 & ...
        ~isnan(tmpSC) & ~isinf(tmpSC) & tmpSC~=0;
    GD_tri = tmpGD(keep);
    SC_tri = tmpSC(keep);
    FC_tri = tmpFC(keep);
    MPC_tri = tmpMPC(keep);
    
    % Predictors
    X = {};
    X{1} = GD_tri;
    X{2} = [GD_tri, SC_tri];
    X{3} = [GD_tri, FC_tri];
    X{4} = [GD_tri, MPC_tri];
    X{5} = [GD_tri, SC_tri, FC_tri];
    X{6} = [GD_tri, SC_tri, MPC_tri];
    X{7} = [GD_tri, FC_tri, MPC_tri];
    X{8} = [GD_tri, SC_tri, FC_tri, MPC_tri];
    
    % Response
    dist_tri = triu(DE,1);
    dist_tri = dist_tri(:);
    dist_tri = dist_tri(keep);
    y = dist_tri;
    
    % save
    DE_pred_boot = {};
    DE_MSE_boot = {};
    DE_R2_boot_train = {};
    DE_R2_boot_test = {};
    
    for this_model = 1:size(X,2)
        
        thisX = X{this_model};
        
        for boot = 1:nrep
            sample = randperm(length(y));
            train = sample( 1 : floor(length(y)*split) );
            test = sample( 1 + (floor(length(y)*split)) : end);
                        
            this_train_X = thisX(train,:);
            this_train_y = y(train);
            
            this_test_X = thisX(test,:);
            this_test_y = y(test);
            
            % fit regression            
            mdl = fitlm(this_train_X,this_train_y);
            ypred = predict(mdl,this_test_X)';
            
            % predicted y in test set
            DE_pred_boot{this_model}(boot,:) = ypred;
            
            % MSE 
            DE_MSE_boot{this_model}(boot,1) = mean((this_test_y - ypred') .^2);
            
            % adjusted r2 in training set
            DE_R2_boot_train{this_model}(boot,1) = mdl.Rsquared.Adjusted;
            
            % Adjusted R2 in test set using parameters estimated from
            % training set
            SSE = sum((this_test_y - ypred') .^2);
            SST = sum((this_test_y - mean(this_test_y)) .^2);
            n = length(this_test_y);
            p = size(mdl.Coefficients,1);
            DE_R2_boot_test{this_model}(boot,1) = 1 - ( ((n-1)/(n-p)) * (SSE/SST));
            
        end
    end
    
end


%% Multilinear model fit for each region seperately

% Baseline, partial and full models: DE ~ b0 + b1*GD + (b2*SC + b3*FC + b4*MPC)
% Note: edges are filtered across all modalities (common edges for all)
for baseline_partial_full_models = 1
    
    DE_pred_reg_partial = {};
    DE_R2_reg_partial = zeros(size(DE,1),8);
    DE_T_reg_partial = {};
    
    for ii = 1:size(DE,1)
        keep = 1:size(DE,1); keep(ii) = [];
        
        % Predictors
        x1 = GDMR(ii,:)';
        x1(ii) = [];
        x2 = SCMR(ii,:)';
        x2(ii) = [];
        x3 = FCMR(ii,:)';
        x3(ii) = [];
        x4 = MPCMR(ii,:)';
        x4(ii) = [];
        
        X = {};
        X{1} = x1;
        X{2} = [x1, x2];
        X{3} = [x1, x3];
        X{4} = [x1, x4];
        X{5} = [x1, x2, x3];
        X{6} = [x1, x2, x4];
        X{7} = [x1, x3, x4];
        X{8} = [x1, x2, x3, x4];
                
        % Response
        y = DE(ii,:)';
    	y(ii) = [];

        for jj = 1:size(X,2)
            thisX = X{jj};
            mdl = fitlm(thisX,y);
            % Uncomment the next line to save region specific model
            % parameters
            %save([outdir, 'node_specific_models/region_',char(string(ii)),'_mdl_',char(string(jj)),'.mat'],'mdl');

            ypred = predict(mdl,thisX)';
            tmp = zeros(1,size(DE,1)); tmp(keep) = ypred;
            DE_pred_reg_partial{jj}(ii,:) = tmp;
            DE_R2_reg_partial(ii,jj) = mdl.Rsquared.Adjusted;
            DE_T_reg_partial{jj}(ii,:) = mdl.Coefficients.tStat';
        end
        
    end 
    
end


%% Model-specific gains in r2 across regions

% r2 difference from more complex models over GD only model
diffs = [];
for ii = 2:size(DE_R2_reg_partial,2)
    diffs(:,ii-1) = DE_R2_reg_partial(:,ii) - DE_R2_reg_partial(:,1);
end

% Figure: full model - baseline model difference in r2
for figure_time = 1
    
    show = diffs(:,end);
    axisLim = [-0.25 0.25];
    show(isnan(show)) = -1;
    load([utildir, 'parcMask.mat'])
    
    fig = figure, 
    scatter(G1, G2, 120, show, 'filled', ...
            'MarkerEdgeColor', [0 0 0], 'LineWidth', 1); 
    xlim([-1.1, 1.1]); ylim([-1.1 1.1]);
    set(gca, 'colormap', vik); caxis(axisLim)
    axis('square');
    
    fig = figure,
    gFull = zeros(size(show,1), 1);
    gFull(~parcMask,1) = show;
    fParcUp = mica_parcelData2surfData(gFull',c69,parcFS);
    fParcUp(maskParc == 1) = 0;
    data = fParcUp(nFS/2+1:end);
    
    a(1) = axes('position', [0.1 0.45 0.3 0.3]);
    p_left = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',vik,1,axisLim);
    camlight(80,-10);
    camlight(-80,-10);
    view([-90 0])
    axis off
    axis image
    
    a(2) = axes('position', [0.1 0.15 0.3 0.3]);
    p_right = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',vik,1,axisLim);
    camlight(80,-10);
    camlight(-80,-10);
    view([90 0])
    axis off
    axis image

end 


%% Correlate model gains with FC G1

% Load gradient
load([utildir, 'margulies_G1.mat']);
[FCG1_parc, ~] = mica_surfData2parcelData(FC_G1', parcFS);
FCG1_parcR = FCG1_parc(parcMask==0)';

r = [];
for ii = 1:size(diffs,2)
    r(ii) = corr(diffs(:,ii), FCG1_parcR, 'type', 'spearman');
end

% Spin test
[sphere_lh, sphere_rh] = load_conte69('spheres');
n_permutations = 1000;
y_rand = spin_permutations({FC_G1(1:nFS/2), FC_G1(1+nFS/2:end)}, ...
                  {sphere_lh,sphere_rh}, ...
                  n_permutations,'random_state',0);
gRand = [squeeze(y_rand{1}); squeeze(y_rand{2})];

% Correlations with null model patterns to test statistical significance
corrPerm = zeros(n_permutations,length(r));
for ii = 1:n_permutations
    [gRandParc, ~] = mica_surfData2parcelData(gRand(:,ii)', parcFS);
    gRandParc = gRandParc(102:end);
    keep = ~isnan(gRandParc);
    
    corrPerm(ii,:) = corr(gRandParc(keep)', diffs(keep,:), 'type', 'spearman');
end

pval = [];
for ii = 1:length(r)
    pval(ii,1) = length(find(abs(corrPerm(:,ii)) > abs(r(ii))))/n_permutations;
        
    figure,
        r_G = histogram(corrPerm(:,ii)); hold on
        plot(repmat(r(ii),[1,max(r_G.Values)]), 1:max(r_G.Values), 'r:', 'LineWidth', 2); hold off
end


