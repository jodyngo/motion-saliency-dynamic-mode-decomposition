%% clear and close all

clc; clear all; close all;
addpath(genpath('Dependencies'));
addpath(genpath('utilities'));


%% Read frames from baseline

tic
im_path = 'F:\Resources Video\dataset2014\dataset\dynamicBackground\canoe\cut';
out_path = 'F:\Resources Video\dataset2014\dataset\dynamicBackground\canoe\saliencymap';

im_name = imagePathRead(im_path);
im_n = length(im_name);

for k = 1:im_n
    img = imread((fullfile(im_path,im_name{k})));
    im_in = im2double(imread((fullfile(im_path,im_name{k}))));
    im_in = imresize(im_in, 0.5); 
    disp(['Loading Frame number ',num2str(k)]);
    gray = rgb2gray(im_in);
    S(:,k) = gray(:);
end
[m,n] = size(gray);
toc
%%
tic
[LowRank, Sparse] = ComputeSaliencyVideo(S,m,n,im_n);
toc

%% Saliency Map
figure;

tic
disp('Generate Saliency Map');
for j = 1:im_n
    %imshow(Sparse(:,:,j));
    sl_map=generatesalientMap(Sparse(:,:,j)); imshow(sl_map);
    imwrite(sl_map,fullfile(out_path, sprintf(im_name{j}, j)));
end
toc

%% Evaluate saliency map
gtPath = 'GROUND_TRUTH';  
resSalPath = 'SAL_MAP';                     % result path
if ~exist(resSalPath, 'file')
    mkdir(resSalPath);
end

gtSuffix = '.png';
resPath = 'results';
if ~exist(resPath,'file')
    mkdir(resPath);
end

%% compute Precison-recall curve
[rec, pre] = DrawPRCurve(resSalPath, '.png', gtPath, gtSuffix, true, true, 'r');
PRPath = fullfile(resPath, ['PR.mat']);
save(PRPath, 'rec', 'pre');
fprintf('The Precison-Recall curve is saved in the file: %s \n', resPath);

%% compute ROC curve
thresholds = [0:1:255]./255;
[TPR, FPR] = CalROCCurve(resSalPath, '.png', gtPath, gtSuffix, thresholds, 'r');    
ROCPath = fullfile(resPath, ['ROC.mat']);
save(ROCPath, 'TPR', 'FPR');
fprintf('The ROC curve is saved in the file: %s \n', resPath);

%% compute F-measure curve
setCurve = true;
[meanP, meanR, meanF] = CalMeanFmeasure(resSalPath, '.png', gtPath, gtSuffix, setCurve, 'r');
FmeasurePath = fullfile(resPath, ['FmeasureCurve.mat']);
save(FmeasurePath, 'meanF');
fprintf('The F-measure curve is saved in the file: %s \n', resPath);

%% compute MAE
MAE = CalMeanMAE(resSalPath, '.png', gtPath, gtSuffix);
MAEPath = fullfile(resPath, ['MAE.mat']);
save(MAEPath, 'MAE');
fprintf('MAE: %s\n', num2str(MAE'));

%% compute WF
Betas = [1];
WF = CalMeanWF(resSalPath, '.png', gtPath, gtSuffix, Betas);
WFPath = fullfile(resPath, ['WF.mat']);
save(WFPath, 'WF');
fprintf('WF: %s\n', num2str(WF'));

%% compute AUC
AUC = CalAUCScore(resSalPath, '.png', gtPath, gtSuffix);
AUCPath = fullfile(resPath, ['AUC.mat']);
save(AUCPath, 'AUC');
fprintf('AUC: %s\n', num2str(AUC'));

%% compute Overlap ratio
setCurve = false;
overlapRatio = CalOverlap_Batch(resSalPath, '.png', gtPath, gtSuffix, setCurve, '0');
overlapFixedPath = fullfile(resPath, ['ORFixed.mat']);
save(overlapFixedPath, 'overlapRatio');
fprintf('overlapRatio: %s\n', num2str(overlapRatio'));


