
function [D,Z,Objfun,Sparsity,NSRE,Dchange,Cchange,taxis] = SOUP_DILLO_Npar(Y,J,lambda,K,L,alpha)

global DiffCZs;
global x;
global Z_startN2;
global initDN2;
% global alt;
global method;
% global DL;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
% alpha=1.0;
% Dt=0;Ct=0;
% niter=600;
% Initial Estimates:
% D = genODCT(n,J);
% D=gpuArray(x);
D = initDN2;
% Z = gpuArray(zeros(J,N));
Z=gpuArray(Z_startN2);
Objfun=gpuArray(zeros(1,K+1));
Sparsity=gpuArray(zeros(1,K+1));
NSRE=gpuArray(zeros(1,K+1));
Dchange=gpuArray(zeros(1,K+1));
Cchange=gpuArray(zeros(1,K+1));
taxis=gpuArray(zeros(1,K+1));
Y=gpuArray(Y);
D_old=D;
C_old=Z;
V1=gpuArray(repmat(v1,1,size(D,2)));
% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
%+ Show progress
fprintf(['Running New DL-L',num2str(method),' algorithm\n']);

    if method == 0
        Objfun(1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z));  %L0
    else
        Objfun(1)=norm((Y-D*Z),'fro')^2+((lambda^2)*sum(sum(abs(Z)))); %L1
    end
        Sparsity(1)=nnz(Z)/(n*N);
        NSRE(1)=norm((Y-D*Z),'fro')/reg;
        Dchange(1)=norm((D-D_old),'fro');
        Cchange(1)=norm((Z-C_old),'fro')/reg;
        
        
    lam = alpha*max(eig(mtimes(D',D)))+(1-alpha)*J;
    H_lambda = @(b) b.*(abs(b) >= (lambda/sqrt(lam)));
    DtY = mtimes(D',Y);
    M = (gpuArray(eye(J))-mtimes(D',D/lam));
for t = 2:K+1
    D_old=D;
    C_old=Z';
    %+ Show progress
%     if(t==1)
        fprintf('iteration %2g \n Z update',t-1);
        tic;

        Bt=mtimes(M,Z)+DtY/lam;
        
    if method == 0    
%         L0 Norm
        Z = sign(Bt).*min(abs(H_lambda(Bt)),L);
    else
%         L1 Norm
        Z = sign(Bt).*max(abs(Bt)-((lambda^2)/(2*lam)),0);
    end
%     end
        
    if method == 0
        Objfun(t)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z)); %L0
    else
        Objfun(t)=norm((Y-D*Z),'fro')^2+((lambda^2)*sum(sum(abs(Z))));  %L1 
    end
        if Objfun(t-1)<Objfun(t)
            keyboard
        end
        Sparsity(t)=nnz(Z)/(n*N);
        NSRE(t)=norm((Y-D*Z),'fro')/reg;
        Dchange(t)=norm((D-D_old),'fro');
        Cchange(t)=norm((Z'-C_old),'fro')/reg;
        taxis(t)=taxis(t-1)+toc;

        fprintf('\n');
            
        
 end
%     fprintf('\n');

    Objfun=gather(Objfun);    
    Sparsity=gather(Sparsity);
    NSRE=gather(NSRE);
    Dchange=gather(Dchange);
    Cchange=gather(Cchange);
    taxis=gather(taxis);
    D=gather(D);
    Z=gather(Z);
        %+ Show progress
        
    
        
end