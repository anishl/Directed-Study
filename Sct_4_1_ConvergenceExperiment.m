% Sct_4_1_ConvergenceExperiment  Reproduce convergence experiment results.
%   Sct_4_1_ConvergenceExperiment reproduces the convergence experiment
%   results from Section 4.1 of the arXiv paper: 1511.06333v3
%     S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, "Efficient Sum
%     of Sparse Outer Products Dictionary Learning (SOUP-DIL)," 2016.

%   Author: David Hong (dahong@umich.edu)
%   Last edited: 1 June 2016

%% Setup and Parameters
clear; close all; rng(0);
% tic
% Image/Patch Parameters
fnames = { ...
    'images/barbara.png', ...
    'images/boat.png', ...
    'images/hill.png'
    };
sqrtn = 8; n = sqrtn*sqrtn;
N = 3e4;

% SOUP-DIL[LO] Parameters
J = 256;
lambda = 8;
K = 700;
L = Inf;

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
global x;
x = genODCT1(n,J);

%% Perform experiment
% Run SOUP-DIL[LO]
[D,Z,ObjFunc,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILLO_Npar4(Y,J,lambda,K,L);

% Compute performance metrics
% ObjFunc  = cellfun(@(D,C) norm(Y-D*C','fro')^2 + lambda^2 * nnz(C),Dt,Ct);
% Sparsity = cellfun(@(D,C) nnz(C)/(n*N)                            ,Dt,Ct);
% NSRE     = cellfun(@(D,C) norm(Y-D*C','fro')/norm(Y,'fro')        ,Dt,Ct);
% Dchange  = cellfun(@(Dt,Dt1) norm(Dt - Dt1,'fro')              ,Dt(2:end),Dt(1:end-1));
% Cchange  = cellfun(@(Ct,Ct1) norm(Ct - Ct1,'fro')/norm(Y,'fro'),Ct(2:end),Ct(1:end-1));
% toc
% save(mfilename,'-v7.3');

%% Plot results
figure(3);

% Fig. 3 (a): Objective function
subplot(1,4,1); plot(1:K+1,ObjFunc,'r'); xlim([1 K+1]);
xlabel('Iteration Number'); ylabel('Objective Function');

% Fig. 3 (b): NSRE (percentage) and sparsity factor of C (as a percentage)
subplot(1,4,2); [yyAxes,yySparsity,yyNSRE] = plotyy(1:K+1,100*Sparsity,1:K+1,100*NSRE);
xlim([1 K]); yySparsity.LineStyle = '-'; yyNSRE.LineStyle = '--';
xlabel('Iteration Number'); ylabel(yyAxes(1),'Sparsity (%)'); ylabel(yyAxes(2),'NSRE (%)');
legend('Sparsity (%)','NSRE (%)');

% Fig. 3 (c): Changes between successive D iterates (||Dt - Dt-1||_F)
subplot(1,4,3); semilogy(1:K+1,Dchange,'r-'); xlim([1 K+1]);
xlabel('Iteration Number'); ylabel('||D^t - D^{t-1}||_F');

% Fig. 3 (d): Normalized changes between successive C iterates (||Ct - Ct-1||_F/||Y||_F)
subplot(1,4,4); semilogy(1:K+1,Cchange,'r-'); xlim([1 K+1]);
xlabel('Iteration Number'); ylabel('||C^t - C^{t-1}||_F / ||Y||_F');
