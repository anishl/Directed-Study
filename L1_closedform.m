% 

%% Setup and Parameters
clc
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
J = n;
lambda = 31;
K = 20;
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
%% Solve for Coefficents Z (orthonormal D)

% Performing DCT on collected patches (reshaping the patches back to 3D and back again)
% Yd = reshape(dct3(reshape(Y, sqrtn,sqrtn,[])),[],N);

% Dx=genODCT1(n,n);
% Yd = Dx'*Y;

Yd=dct(Y);

% Find Coefficients
Z = sign(Yd).*max(abs(Yd)-(0.5*(lambda^2)),0);
%% Comparisons and Results

% DCT for D*Z
% Yrec = reshape(idct3(reshape(Z, sqrtn,sqrtn,[])),[],N);
% Yrec = Dx*Z;

Yrec=idct(Z);

% ERROR  and Sparsity
Error = norm(Y-Yrec,'fro')^2 + ((lambda^2)*sum(sum(abs(Z))));
NRSE = norm(Y-Yrec,'fro')/norm(Y,'fro');
Sparsity= nnz(Z)/(n*N);

% ||Cn-Z||_F
global x;
global DiffCZs;
global DiffCZp;
x=Z;
tic
[D_ext,C_ext,Objfun_ext,Sparsity_ext,NSRE_ext,Dchange_ext,Cchange_ext] = SOUP_DILL1_s_CUp(Y,J,lambda,K,L);
disp('DILL1 ')
toc
tic
[D_p,C_p,Objfun_p,Sparsity_p,NSRE_p,Dchange_p,Cchange_p] = SOUP_DILLO_pCUp(Y,J,lambda,K,L);
disp('Jpar ')
toc
%% Plots vs n_iter
figure (5)
subplot 121
plot(1:K,DiffCZs,'-x');
title('SOUP DILL1');
xlabel('iterations')
ylabel('||C-Z^T||_F')
subplot 122
plot(1:K,DiffCZp,'-o');
title('J-Parallel Algo');
xlabel('iterations')
ylabel('||C-Z^t||_F')

% Plots of Convergence for Serial and Parallel
% figure(6);
% 
% % Fig. 3 (a): Objective function
% subplot(1,4,1); plot(1:K,Objfun_ext,'r'); xlim([1 K]);hold on
%                 plot(1:K,Objfun_p,'b'); xlim([1 K]);hold off
% xlabel('Iteration Number'); ylabel('Objective Function');
% 
% % Fig. 3 (b): NSRE (percentage) and sparsity factor of C (as a percentage)
% subplot(1,4,2); [yyAxes_ext,yySparsity_ext,yyNSRE_ext] = plotyy(1:K,100*Sparsity_ext,1:K,100*NSRE_ext);
%                 xlim([1 K]); yySparsity_ext.LineStyle = '-'; yyNSRE_ext.LineStyle = '--';
%                              yySparsity_ext.Color = 'red'; yyNSRE_ext.Color = 'red';
%                 hold on
%                 [yyAxes_p,yySparsity_p,yyNSRE_p] = plotyy(1:K,100*Sparsity_p,1:K,100*NSRE_p);
%                 xlim([1 K]); yySparsity_p.LineStyle = '-'; yyNSRE_p.LineStyle = '--';
%                              yySparsity_p.Color = 'blue'; yyNSRE_p.Color = 'blue';
% xlabel('Iteration Number'); ylabel(yyAxes_ext(1),'Sparsity (%)'); ylabel(yyAxes_ext(2),'NSRE (%)');
% %                             ylabel(yyAxes_p(1),'Sparsity_parallel (%)'); ylabel(yyAxes_p(2),'NSRE_parallel (%)');
% legend('Sparsity serial (%)','NSRE serial(%)','Sparsity parallel (%)','NSRE parallel(%)');
% 
% % % Fig. 3 (c): Changes between successive D iterates (||Dt - Dt-1||_F)
% % subplot(1,4,3); semilogy(1:K,Dchange,'r-'); xlim([1 K]);
% % xlabel('Iteration Number'); ylabel('||D^t - D^{t-1}||_F');
% % 
% % % Fig. 3 (d): Normalized changes between successive C iterates (||Ct - Ct-1||_F/||Y||_F)
% % subplot(1,4,4); semilogy(1:K,Cchange,'r-'); xlim([1 K]);
% % 
