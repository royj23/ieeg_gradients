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


%% Compute PSD and normalize

% NOTE: timeseries file is too large to include in Github repo, but it
% freely available (see README)
% Below is the code (commented) to generate the PSDs from this data file.
% Below, we directly load precomputed PSDs (included in repo)
for compute_and_normalize_psd = 1
    
%     % Load iEEG timeseries
%     load([datadir, 'ieeg_ts.mat']);
% 
%     % psd
%     f = 0.5:0.5:80;
%     fs = 200;
%     pxx = [];
%     for ii = 1:size(Data,2)
%         tmp = Data(:,ii);
%         tmp(tmp == 0) = [];
%         pxx(:,ii) = pwelch(tmp,2*fs,1*fs,f,fs);
%     end
%     
%     % normalize psd to total power = 1
%     pxx_norm = [];
%     for ii = 1:size(pxx,2)
%         pxx_norm(:,ii) = pxx(:,ii)/sum(pxx(:,ii));
%     end
%     
%     % Log transform
%     pxx_norm = log(pxx_norm);
% 
end

% Directly load PSDs
f = 0.5:0.5:80;
fs = 200;
load([datadir, 'pxx_norm.mat']);

% Get mean PSD over all channels
psdMeanAll = mean(pxx_norm,2);


%% Average PSD at vertices covered by overlapping patches

for avrg_and_show_coverage = 1
    
    pxx_avrg = zeros(size(pxx_norm,1), nFS/2);
    for ii = 1:nFS/2

        % Find channels that overlap a given vertex
        patchesOverVertex = find(allChannelsPatches_C69(:,ii) > 0);
        
        % Average those channels
        if ~isempty(patchesOverVertex)
            pxx_avrg(:,ii) = mean(pxx_norm(:,patchesOverVertex),2);
        else
            pxx_avrg(:,ii) = NaN;
        end
    end
    
end


%% Map PSDs to parcellation 

for weighted_average = 1
    
    pxx_parc_all = zeros(size(pxx_norm,1), length(uparc));

    for ii = 1:length(uparcR) % only right hemi is mapped
        this_parcel = uparcR(ii);
        this_parcel_idx = find(parcFSR == this_parcel);
        findChannels = allChannelsPatches_C69(:,this_parcel_idx);
        nVertPerChannel = sum(findChannels,2); % for each channel, how many associated vertices in that parcel
        nVertPerChannelIdx = find(nVertPerChannel > 0);
        nVertPerChannel_tmp = nVertPerChannel(nVertPerChannelIdx);

        if sum(nVertPerChannel) > 0
            all_pxx_parcel = zeros(size(pxx_norm,1), length(nVertPerChannelIdx));
            weights = zeros(1, length(nVertPerChannelIdx));

            for jj = 1:length(nVertPerChannelIdx)
                all_pxx_parcel(:,jj) = pxx_norm(:,nVertPerChannelIdx(jj));
                weights(jj) = nVertPerChannel_tmp(jj);
            end
            
            pxx_parc_all(:,uparcR(ii)+1) = sum((all_pxx_parcel .* weights),2) ./ sum(nVertPerChannel_tmp); % weighted average

        else
            pxx_parc_all(:,uparcR(ii)+1) = NaN;
        end
    end
end


%% Gradient of PSD similarity

% Apply mask
parcMask = sum(pxx_parc_all) == 0;
parcMask(isnan(sum(pxx_parc_all))) = 1;
parcMask(uparc == maskCol) = 1;
fParcMask = pxx_parc_all(:,~parcMask);

% Correlate
fParcCorr = partialcorr(fParcMask,psdMeanAll);
fParcZ = .5 * log( (1+fParcCorr) ./ (1-fParcCorr) );
fParcZ(isinf(fParcZ)) = 1;
fParcZ(find(eye(size(fParcZ)))) = 0;

% Fit gradients
nG = size(fParcZ,1)-1;
T = 90;

gm = GradientMaps('kernel', 'na', 'alignment', 'pa', 'approach', 'dm', 'n_components', nG);
psdG = gm.fit(fParcZ, 'sparsity', T);
G = psdG.gradients{1};

% Keep gradients up to a cumulated 50% variance
nG_keep = cumsum(psdG.lambda{1} ./ sum(psdG.lambda{1}));
nG_keep = double(nG_keep > 0.5); nG_keep = find(nG_keep,1,'first');


%% Show first two gradients 

G1 = -rescale(G(:,1),-1,1); G2 = -rescale(G(:,2),-1,1);
showG = [G1,G2];
gCol = colour2gradients(G1,G2);

% Show G1 and G2
for this_figure = 1
    c_range = [-1 1];

    for ii = 1:size(showG,2)
    
        fig = figure,

        % Add Inf at mask data indices
        gFull = zeros(size(pxx_parc_all,2), 1);
        gFull(~parcMask,1) = showG(:,ii);
        
        fParcUp = mica_parcelData2surfData(gFull',c69,parcFS);
        fParcUp(maskParc == 1) = 0;
        data = fParcUp(nFS/2+1:end);
        
        % Plot
        a(1) = axes('position', plot_brains(2,:));
        p_left = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',roma,1,c_range);
        camlight(80,-10);
        camlight(-80,-10);
        view([-90 0])
        axis off
        axis image
        
        a(2) = axes('position', plot_brains(1,:));
        p_right = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',roma,1,c_range);
        camlight(80,-10);
        camlight(-80,-10);
        view([90 0])
        axis off
        axis image
    
    end

    % Project new colorspace to cortical surface
    gFull = zeros(size(pxx_parc_all,2), 1);
    gFull(~parcMask,1) = 1:size(gCol,1);
    fParcUp = mica_parcelData2surfData(gFull',c69,parcFS);
    fParcUp(maskParc == 1) = 0;
    cmap = [0.5 0.5 0.5; gCol];
    data = fParcUp(nFS/2+1:end);

    fig = figure,
    a(1) = axes('position', plot_brains(2,:));
    p_left = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',cmap,1);
    camlight(80,-10);
    camlight(-80,-10);
    view([-90 0])
    axis off
    axis image
    
    a(2) = axes('position', plot_brains(1,:));
    p_right = plotSurfaceROIBoundary(c69R,parcFSR,data,'midpoint',cmap,1);
    camlight(80,-10);
    camlight(-80,-10);
    view([90 0])
    axis off
    axis image
    caxis([-1 1])

end

% Scatter plot
figure, scatter(G1, G2, 120, gCol, 'filled', ...
                'MarkerEdgeColor', [0 0 0]); 
        xlim([-1.1, 1.1]);
        ylim([-1.1 1.1]);
        axis('square')


%% Save data for next step

cd(datadir)
save gradients-schaefer-200 G G1 G2 nG_keep 