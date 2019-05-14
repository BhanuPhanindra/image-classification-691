function [features, dict] = computeFeatures(imdb, param)
% COMPUTEFEATURES computes features on all images in the set.
%   [FEATURES, DICT] = COMPUTEFEATURES(IMDB, PARAM) comptutes features on
%   all the images in IMDB.IMAGES.NAMES using paramters specified in PARAM.
%   if there are N images, the output FEATURE is a array of size [D N],
%   where D is the feature dimension. In addition a DICT structure is also
%   returned which is the dictionary learned for bag of words model, i.e.,
%   when PARAM.FEATURE = {'bow-sift', 'bow-patches'}, it is non-empty, but
%   is empty otherwise. By default it computes random features.
%
% This code is developed by Subhransu Maji from UMass Amherst
%
% Homework 3: Recognition

switch param.feature
    case 'tinyimage'
        features = tinyimageFeatures(imdb, param);
        dict = [];
    case 'gist'
        features = gistFeatures(imdb, param);
        dict = [];
    case 'bow-sift'
        [features, dict] = bowFeaturesSIFT(imdb, param);
    case 'bow-patches'
        [features, dict] = bowFeaturesPatches(imdb, param);
    otherwise
        disp('warning: using random features?');
        features = randomFeatures(imdb, param);
        dict = [];
end
fprintf('Computed %i %s features.\n', size(features, 2), param.feature);

%--------------------------------------------------------------------------
%                                                           random features
%--------------------------------------------------------------------------
function features = randomFeatures(imdb, param)
numImages = length(imdb.images.name);
features = rand(3, numImages);

%--------------------------------------------------------------------------
%                                                       tiny image features
%--------------------------------------------------------------------------
function features = tinyimageFeatures(imdb, param)
% Compute grayscale features by resizing the images into a pathDim x patchDim 
% patch and resizing the image into a vector.

% Implement this
numImages = length(imdb.images.name);
dim = param.tinyimage.patchDim;
features = zeros(dim * dim * 3, numImages);
for i = 1:numImages
    f = fullfile(imdb.imageDir, imdb.images.name{i});
    I = imread(f);
    I = imresize(I, [dim, dim]);
    I = reshape(I, [dim * dim * 3, 1]);
    features(:, i) = I;
    
end

    

%--------------------------------------------------------------------------
%                                                   gist feature descriptor
%--------------------------------------------------------------------------
function features = gistFeatures(imdb, param)
% 
% Optionally implement this


%--------------------------------------------------------------------------
%                                                   bag of words model with 
%                                                   dense local pathces
%--------------------------------------------------------------------------
function [features, dict] = bowFeaturesPatches(imdb, param)

% STEP 1: Write a function that extracts dense grayscale patches from an image
% STEP 2: Learn a dictionary
%           -- sample many desriptors (~10k) from train+val images
%           -- learn a dictionary using k-means
% STEP 3: Loop over all the images in imdb.images.names and extract
%         features (same as step 1) and assign them to dictionary items.
%         Build global histograms over these.
%
% Some useful code snippets:
%
%   trainValId = find(ismember(imdb.images.imageSet, [1 2]));
%   testId = find(ismember(imdb.images.imageSet, 3));
%  
%   D = dist2(dict, localFeatures); % compute all pair distances 
%   [~, codeWord] = min(D, [], 1); % compute codeword assignment
%   NOTE: make sure the dimensions of dict and localFeatures are correct.
%
%   wordHist = histc(codeWord, 1:dictionarySize); % build histogram

% Implement this
features = [];
dict = [];

%--------------------------------------------------------------------------
%                                             bag of words model with dense
%                                             SIFT descriptors
%--------------------------------------------------------------------------
function [features, dict] = bowFeaturesSIFT(imdb, param)
% STEP 1: Write a function that extracts dense SIFT features from an image
% STEP 2: Learn a dictionary
%           -- sample many desriptors (~10k) from train+val images
%           -- learn a dictionary using k-means
% STEP 3: Loop over all the images in imdb.images.names and compute
%         features (same as step 1) and assign them to dictionary items.
%
% Some useful code snippets:
%
%   trainValId = find(ismember(imdb.images.imageSet, [1 2]));
%   testId = find(ismember(imdb.images.imageSet, 3));
%  
%   D = dist2(dict, localFeatures); % compute all pair distances 
%   [~, codeWord] = min(D, [], 1); % compute codeword assignment
%   NOTE: make sure the dimensions of dict and localFeatures are correct.
%
%   wordHist = histc(codeWord, 1:dictionarySize); % build histogram

% Implement this
features = [];
dict = [];

