%% Setup and Parameters
clear; close all; rng(0);

% Image/Patch Parameters
fnames = { ...
    'images/barbara.png', ...
    'images/boat.png', ...
    'images/hill.png'
    };
sqrtn = 8; n = sqrtn*sqrtn;
N = 3e4;
% SOUP-DIL[LO] Parameters
% J = 256;
lambda = 30;
% K = 80;
% L = Inf;

%% Load images and extract patches
% Load images
images = cellfun(@(fname) im2double(imread(fname)),fnames,'UniformOutput',false);
images = cellfun(@(image) image*255,images,'UniformOutput',false);

% Extract all patches
patches = cellfun(@(image) im2col(image,[sqrtn sqrtn],'sliding'),images,'UniformOutput',false);
patches = cell2mat(patches);

% Randomly select subset of patches
subset = randperm(size(patches,2)); subset = subset(1:N);
Y = patches(:,subset);
D=genODCT(n,n);
[X,ObjFunc,Sparsity,NSRE]=DILL1_test(Y,D,lambda);

