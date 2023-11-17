clc;clear;
close all;

% Add path of the video file
addpath(genpath('Dependencies'));

%% Path settings
% ====== For video frames ======
 inputImgPath = 'INPUT_IMG';                 % input image path
 resSalPath = 'SAL_MAP';                     % result path

%% Calculate saliency using DMD
imgFiles = imdir(inputImgPath);

skip_frames = 1;
for indImg = 1:skip_frames:length(imgFiles)       
    % read image
    imgPath = fullfile(inputImgPath, imgFiles(indImg).name);
    img.RGB = imread(imgPath);
    img.name = imgPath((strfind(imgPath,'\')+1):end);
    %calculate saliency map via DMD
    tic;
    salMap = ComputeSaliency(img);
    toc;
    if(mod(indImg, 1) == 0)
        sal_figure = figure(2);
        subplot(1,2,1);imshow(img.RGB,[]);title('Input RGB Image');
        subplot(1,2,2);imshow(salMap,[]);title('Saliency Map of DMD Method');
        pause;
        close(sal_figure);
    end
        
    % save saliency map    
    %salPath = fullfile(resSalPath, strcat(img.name(1:end-4), '.png'));  
    %imwrite(salMap,salPath);
    imwrite(salMap,fullfile(Folder, sprintf('%06d.jpg', indImg)));
    %imwrite(salMap,'salMap-7586-min.png');
    %fprintf('The saliency map: %s is saved in the file: SAL_MAP \n', img.name);
    %fprintf('%s/%s images have been processed ... Press any key to continue ...\n', num2str(indImg), num2str(length(imgFiles)) );   
    close all;
end

%% Evaluation

