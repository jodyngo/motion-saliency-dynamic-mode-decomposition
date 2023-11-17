% SDMD_RUN An example to demonstrate incrementally updated DMD.
% FOR VIDEO
% set example parameters here
clc; clear all; close all;
addpath(genpath('DMD'));
addpath(genpath('Dependencies'));

im_path = 'F:\Resources Video\dataset2014\dataset\baseline\highway\cut';
out_path = 'F:\Resources Video\dataset2014\dataset\baseline\highway\saliencymap';
im_name=imagePathRead(im_path);

max_rank = 1;        % maximum allowable rank of the DMD operator
m = length(im_name);  % total number of snapshots to be processed
Path = fullfile(im_path,im_name{1});
[h,w,c] = size(imresize(imread(Path),0.5));
n = h*w;              % number of states
streaming = true;

%% define the example system

% sampling time
dt = 9/m;


%% collect the data

disp('Collecting data')

if ~streaming
    % standard algorithm: batch-processing
    X = zeros(n, m);
    Y = zeros(n, m);
    im_in = im2double(imread((fullfile(im_path,im_name{1}))));
    yk = rgb2gray(im_in);
    for k = 1:m        
        xk = yk;
        yk = rgb2gray(im2double(imread((fullfile(im_path,im_name{k})))));
        X(:,k) = xk(:);
        Y(:,k) = yk(:);
    end
    [Qx, S, V] = svd(X, 0);
    Ktilde = Qx' * Y * V * pinv(S);
    [EigVec, EigVal] = eig(Ktilde);
    Psi = ((X2*V)/E)*EigVec; %DMD Modes %Psi = U * EigVec;
    OmegaExp = exp(log(EigVal)); %complex
    Fourierfreq = abs(log(diag(EigVal)));
    b = pinv(Psi)*X1(:,1);
    
   LowRankFreq = exp(min(Fourierfreq));
   XDMD = zeros(h * w, m);
   XLow = zeros(h * w, m);
   
   for t = 1:m
        XDMD(:, t) = Psi * OmegaExp.^t * b;
   end
   for t = 1:m
        XLow(:, t) = Psi * LowRankFreq.^t * b;
   end
   
    XLow = abs(XLow);
    XSparse = abs(XDMD - XLow);
    XDMD = abs(XDMD);
    Sparse = reshape(XSparse, [h, w, m]);
    LowRank = reshape(XLow, [h, w, m]);

    for j = 1:m
        imshow(Sparse(:,:,j));        
    end
else
    
   % streaming algorithm
    sdmd = StreamingDMD(max_rank);
    im_in = im2double(imread((fullfile(im_path,im_name{1}))));
    yk = imresize(im_in,0.5);
    for k = 1:m
        xk = yk;
        yk = imresize(im2double(imread((fullfile(im_path,im_name{k})))),0.5);        
        
        X{1}(:,k)=reshape(xk(:,:,1),1,[]);
        X{2}(:,k)=reshape(xk(:,:,2),1,[]);
        X{3}(:,k)=reshape(xk(:,:,3),1,[]);
        
        X{1}(:,k+1)=reshape(yk(:,:,1),1,[]);
        X{2}(:,k+1)=reshape(yk(:,:,2),1,[]);
        X{3}(:,k+1)=reshape(yk(:,:,3),1,[]);

        D = [X{1};X{2};X{3}];
        Dx = D(:,k);
        Dy = D(:,k+1);
        
        tic
        sdmd = sdmd.update(Dx, Dy); 
        [modes, evalsD, evals] = sdmd.compute_modes();
        %clear X;
        %clear D;
        
        % generate Background
        Psi = modes;
        OmegaExp = exp(log(evals)); 
        Fourierfreq = abs(log(evalsD));
        LowRankFreq = exp(min(Fourierfreq));
        
        [~, index] = min(Fourierfreq);
        
        DMS = Psi(:,index);
        R = abs(reshape(DMS(1:h*w,1),h,w));
        G = abs(reshape(DMS(h*w+1:2*h*w,1),h,w));
        B = abs(reshape(DMS(2*h*w+1:end,:),h,w));
        BGImage = cat(3, mat2gray(R),mat2gray(G),mat2gray(B)); %imshow(BGImage); 
        FGImage1 = xk - BGImage; imshow(FGImage1);
        %imwrite(FGImage1,fullfile(out_path, sprintf(im_name{k}, k)));
%         b = pinv(Psi)*Dy;
%         XDMD(:, :) = Psi * OmegaExp * b;  
%         XLow(:, :) = Psi * LowRankFreq * b; 
% 
%         XLow = abs(XLow);
%         XSparse = abs(XDMD - XLow);
%         SparseR = abs(reshape(XSparse(1:h*w,1),h,w));
%         SparseG = abs(reshape(XSparse(h*w+1:2*h*w,1),h,w));
%         SparseB = abs(reshape(XSparse(2*h*w+1:end,:),h,w));
%         FGImage2 = cat(3, mat2gray(SparseR),mat2gray(SparseG),mat2gray(SparseB));
%         imshow(FGImage2);
        
        
        
        %sl_map=generatemotionsalientMap(FGImage1); imshow(sl_map);   
        
        toc
        
    end
end
elapsed_time = toc;
fprintf('  Elapsed time: %f seconds\n', elapsed_time); 


%% compute DMD spectrum
disp('Computing spectrum')
tic
if ~streaming
    % standard DMD spectrum
    [evecK, evals] = eig(Ktilde);
    evalsD = diag(evals);
    modes = Qx * evecK; %Psi
else
   % streaming algorithm
    [modes, evalsD, evals] = sdmd.compute_modes();
end
elapsed_time = toc;
fprintf('  Elapsed time: %f seconds\n', elapsed_time)

% calculate corresponding frequencies
fdmd = abs(angle(evalsD)) ./ (2 * pi * dt);
ydmd = zeros(length(fdmd),1);
for ii = 1:length(fdmd)
    ydmd(ii) = norm(modes(:,ii)).*abs(evalsD(ii));
end
ydmd = ydmd./max(ydmd);

figure(1)
stem(fdmd,ydmd,'o-')
xlabel('Frequency')
ylabel('Magnitude')

