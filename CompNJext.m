% Setup and Parameters
set(0, 'DefaultAxesLineWidth', 2.0)
set(0, 'DefaultTextFontSize', 18)
set(0, 'DefaultTextFontWeight', 'normal')
set(0, 'DefaultAxesFontSize', 18)
%set(0, 'DefaultAxesFontWeight', 'bold')
set(0, 'DefaultAxesFontWeight', 'normal')
set(0, 'DefaultLineMarkerSize', 10)
reset(gpuDevice);
rng(0);
% clear; close all; rng(0);
% tic
% Image/Patch Parameters
fnames = { ...
    'images/barbara.png', ...
    'images/boat.png', ...
    'images/hill.png'
    };
sqrtn = 8; n = sqrtn*sqrtn;
N = 6e4;
global alt;
global method;
global DL;
DL = 1; % 1 for yes, 0 for no Dictionary Update
% method = 0; %L :  1 or 0
center = 1; % Indicates whether the dictionary atoms are centered (1) or not (0).
% SOUP-DIL[LO] Parameters
% J=n;
J = 256;
global lambda;
lambda = 10; % For when not testing parameters.
K1 = 20;
K2 = 40;% 40 % SOUP DIL Iterations
K3=479;% 179 % Npar Iterations
K4=1;
L = Inf;
% alt=20; % 60 % iterations between updating the dictionary
K2=10*K2;K3=10*K3;
% Load images and extract patches
% Load images
images = cellfun(@(fname) im2double(imread(fname)),fnames,'UniformOutput',false);
images = cellfun(@(image) image*255,images,'UniformOutput',false);

% Extract all patches
patches = cellfun(@(image) im2col(image,[sqrtn sqrtn],'sliding'),images,'UniformOutput',false);
patches = cell2mat(patches);

% Randomly select subset of patches
subset = randperm(size(patches,2)); subset = subset(1:N);
Y = patches(:,subset);

global x ;
x = (genODCT1(n,J,center));

%% For when Nseq is not in use
 Sparsity_Nseq=0; NSRE_Nseq=0;
%% Run JparL0
% [DJ,ZJ,ObjFuncJ,SparsityJ,NSREJ,DchangeJ,CchangeJ] = SOUP_DILLO_pCUp(Y,J,lambda,K,L);

%% Run NparL0 [variables have ext: N]
% alpha=[0 1.0];
alpha=1.0;
DN=[];ZN=[];ObjFuncN=[];SparsityN=[];NSREN=[];DchangeN=[];CchangeN=[];taxisN=[];
for i=1:length(alpha)
    [DN(:,:,i),ZN(:,:,i),ObjFuncN(i,:),SparsityN(i,:),NSREN(i,:),DchangeN(i,:),CchangeN(i,:),taxisN(i,:)] = SOUP_DILLO_Npar0(Y,J,lambda,K1,L,alpha(i));
end
global Z_start;
Z_start = zeros(size(ZN)); % for all zeros initialization
Z_start=ZN; % for the Npar updated sparse code initialization
% pause(5)
%% Run SOUP-DILL0 [variables have ext: _ext]
reset(gpuDevice);
[D_ext,Z_ext,ObjFunc_ext,Sparsity_ext,NSRE_ext,Dchange_ext,Cchange_ext,taxis_ext] = SOUP_DILLO_s_CUp(Y,J,lambda,K2,L);
reset(gpuDevice);
%% Run Npar again (with Jpar D update) [variables have ext: N2]
DN2=[];ZN2=[];ObjFuncN2=[];SparsityN2=[];NSREN2=[];DchangeN2=[];CchangeN2=[];taxisN2=[];
for i=1:length(alpha)
    [DN2(:,:,i),ZN2(:,:,i),ObjFuncN2(i,:),SparsityN2(i,:),NSREN2(i,:),DchangeN2(i,:),CchangeN2(i,:),taxisN2(i,:)] = SOUP_DILLO_Npar(Y,J,lambda,K3,L,alpha(i));
end
reset(gpuDevice);
%% Run Npar with SOUP D update [variables have ext: N3]
DN3=[];ZN3=[];ObjFuncN3=[];SparsityN3=[];NSREN3=[];DchangeN3=[];CchangeN3=[];taxisN3=[];
for i=1:length(alpha)
    [DN3(:,:,i),ZN3(:,:,i),ObjFuncN3(i,:),SparsityN3(i,:),NSREN3(i,:),DchangeN3(i,:),CchangeN3(i,:),taxisN3(i,:)] = SOUP_DILLO_Npar2(Y,J,lambda,K3,L,alpha(i));
end
reset(gpuDevice);
%% Run SOUP with C update first and then d1,d2,.....,dj [variables have ext: _t]
[D_t,Z_t,ObjFunc_t,Sparsity_t,NSRE_t,Dchange_t,Cchange_t,taxis_t] = SOUP_DILLO_test(Y,J,lambda,K2,L);
reset(gpuDevice);
%% Run Npar C update (15 iter),d1,C update, d2....... C update [variables have ext: _Nseq]
[D_Nseq,Z_Nseq,ObjFunc_Nseq,Sparsity_Nseq,NSRE_Nseq,Dchange_Nseq,Cchange_Nseq,taxis_Nseq] = SOUP_DILLO_Npar_seq(Y,J,lambda,K4,L,alpha);

%% Plot results
% figure(3);
% 
% % Fig. 3 (a): Objective function
% for i=1:length(alpha)
%     plot(1:2*K3+1,ObjFuncN2(i,:)); xlim([1 2*K3+1]); hold on;
% end
% xlabel('Iteration Number'); ylabel('Objective Function'); 
% legend ( 'Npar alpha=1.0');

% figure(5);
% for i=1:length(alpha)
%     plot(1:K+1,DchangeN(i,:)); xlim([1 K+1]); hold on;
% end
% xlabel('Iteration Number'); ylabel('Dchange');

% plot(1:K+1,ObjFuncJ,'b'); xlim([1 K+1]);
% xlabel('Iteration Number'); ylabel('Objective Function'); hold on;

% plot(1:K+1,ObjFunc_ext); xlim([1 K+1]);
% xlabel('Iteration Number'); ylabel('Objective Function'); hold on;

% legend('Npar alpha=0','Npar alpha=0.1','Npar alpha=0.5','Npar alpha=0.8', 'Npar alpha=0.9','Npar alpha=1.0','SOUP-DILLO')

figure(5);
for i=1:length(alpha)
    plot(taxisN2(i,:),ObjFuncN2(i,:),'-o');hold on;
    plot(taxisN3(i,:),ObjFuncN3(i,:),'-o');hold on;
end
plot(taxis_ext,ObjFunc_ext,'-x');hold on
plot(taxis_t,ObjFunc_t,'-x');hold on
% plot(taxis_Nseq,ObjFunc_Nseq,'-.');hold on
xlabel('Time (s)'); ylabel('Objective Function'); 
legend(['Npar/Jpar |(Sprsty,NSRE)= ',num2str(100*SparsityN2(end)), '%,',num2str(100*NSREN2(end)),'% | niter= ',num2str(K3)],... 
    ['Npar/Seq D-up |(Sprsty,NSRE)= ',num2str(100*SparsityN3(end)), '%,',num2str(100*NSREN3(end)),'% | niter= ',num2str(K3)],...
    ['SOUP-DILLO |(Sprsty,NSRE)= ',num2str(100*Sparsity_ext(end)), '%,',num2str(100*NSRE_ext(end)),'% | niter= ',num2str(K2)],...
    ['SOUP-DILLO C,d1,..dj |(Sprsty,NSRE)= ',num2str(100*Sparsity_t(end)), '%,',num2str(100*NSRE_t(end)),'% | niter= ',num2str(K2)],...
    ['Npar Seq |(Sprsty,NSRE)= ',num2str(100*Sparsity_Nseq(end)), '%,',num2str(100*NSRE_Nseq(end)),'% | niter= ',num2str(K1)]);



title(['OC JF-AL Centered = ',num2str(center),' |lambda = ', num2str(lambda),'| J,N=',num2str(J),',',num2str(N),'| D updated every ' ,num2str(alt),' iter for Npar',' method = L',num2str(method)])

figure(6)
for i=1:length(alpha)
    plot(taxisN2(i,:),100*SparsityN2(i,:),'-o');hold on;
    plot(taxisN3(i,:),100*SparsityN3(i,:),'-o');hold on;
end
plot(taxis_ext,100*Sparsity_ext,'-x');hold on
plot(taxis_t,100*Sparsity_t,'-x');hold on
% plot(taxis_Nseq,ObjFunc_Nseq,'-.');hold on
xlabel('Time (s)'); ylabel('Sparsity');
legend(['Npar/Jpar |(Sprsty,NSRE)= ',num2str(100*SparsityN2(end)), '%,',num2str(100*NSREN2(end)),'% | niter= ',num2str(K3)],... 
    ['Npar/Seq D-up |(Sprsty,NSRE)= ',num2str(100*SparsityN3(end)), '%,',num2str(100*NSREN3(end)),'% | niter= ',num2str(K3)],...
    ['SOUP-DILLO |(Sprsty,NSRE)= ',num2str(100*Sparsity_ext(end)), '%,',num2str(100*NSRE_ext(end)),'% | niter= ',num2str(K2)],...
    ['SOUP-DILLO C,d1,..dj |(Sprsty,NSRE)= ',num2str(100*Sparsity_t(end)), '%,',num2str(100*NSRE_t(end)),'% | niter= ',num2str(K2)],...
    ['Npar Seq |(Sprsty,NSRE)= ',num2str(100*Sparsity_Nseq(end)), '%,',num2str(100*NSRE_Nseq(end)),'% | niter= ',num2str(K1)]);
title(['OC JF-AL Centered = ',num2str(center),' |lambda = ', num2str(lambda),'| J,N=',num2str(J),',',num2str(N),'| D updated every ' ,num2str(alt),' iter for Npar',' method = L',num2str(method)])


figure(7)
for i=1:length(alpha)
    plot(taxisN2(i,:),100*NSREN2(i,:),'-o');hold on;
    plot(taxisN3(i,:),100*NSREN3(i,:),'-o');hold on;
end
plot(taxis_ext,100*NSRE_ext,'-x');hold on
plot(taxis_t,100*NSRE_t,'-x');hold on
% plot(taxis_Nseq,ObjFunc_Nseq,'-.');hold on
xlabel('Time (s)'); ylabel('NSRE');
legend(['Npar/Jpar |(Sprsty,NSRE)= ',num2str(100*SparsityN2(end)), '%,',num2str(100*NSREN2(end)),'% | niter= ',num2str(K3)],... 
    ['Npar/Seq D-up |(Sprsty,NSRE)= ',num2str(100*SparsityN3(end)), '%,',num2str(100*NSREN3(end)),'% | niter= ',num2str(K3)],...
    ['SOUP-DILLO |(Sprsty,NSRE)= ',num2str(100*Sparsity_ext(end)), '%,',num2str(100*NSRE_ext(end)),'% | niter= ',num2str(K2)],...
    ['SOUP-DILLO C,d1,..dj |(Sprsty,NSRE)= ',num2str(100*Sparsity_t(end)), '%,',num2str(100*NSRE_t(end)),'% | niter= ',num2str(K2)],...
    ['Npar Seq |(Sprsty,NSRE)= ',num2str(100*Sparsity_Nseq(end)), '%,',num2str(100*NSRE_Nseq(end)),'% | niter= ',num2str(K1)]);

title(['OC JF-AL Centered = ',num2str(center),' |lambda = ', num2str(lambda),'| J,N=',num2str(J),',',num2str(N),'| D updated every ' ,num2str(alt),' iter for Npar',' method = L',num2str(method)])
% legend('Npar alpha=0','Npar alpha=0.1','Npar alpha=0.5','Npar alpha=0.8', 'Npar alpha=0.9','Npar alpha=1.0','SOUP-DILLO')
%% Speed-up Calculation
% conv_tN=taxisN(end,2);
% idx_ext=length(find(ObjFuncN(end,2)<=ObjFunc_ext));
% conv_text=taxis_ext(idx_ext+1);
% 
% speedup=conv_text/conv_tN;
% sprintf('the speedup is %d',speedup)