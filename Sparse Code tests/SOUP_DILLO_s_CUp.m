
function [D,C,Objfun,Sparsity,NSRE,Dchange,Cchange,taxis] = SOUP_DILLO_s_CUp(Y,J,lambda,K,L)

global DiffCZs;
global x;
global Z_start_ext;
global initD_ext;
global method;
% global DL;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = gpuArray(In(:,1));
H_lambda = @(b) b.*(abs(b) >= (lambda));
% Dt=0;Ct=0;

% Initial Estimates:
% D = gpuArray(x);
D = initD_ext;
% C = gpuArray(zeros(N,J));
C = gpuArray(Z_start_ext);
% C = gpuArray(x');
Objfun=gpuArray(zeros(1,K+1));
Sparsity=gpuArray(zeros(1,K+1));
NSRE=gpuArray(zeros(1,K+1));
Dchange=gpuArray(zeros(1,K+1));
Cchange=gpuArray(zeros(1,K+1));
taxis=gpuArray(zeros(1,K+1));
Y = gpuArray(Y);

D_old=D;
C_old=C;
% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
    if method == 0
        Objfun(1)=norm((Y-D*C'),'fro')^2+((lambda^2)*nnz(C)); %L0
    else
        Objfun(1)=norm((Y-D*C'),'fro')^2+((lambda^2)*sum(sum(abs(C)))); %L1
    end
    Sparsity(1)=nnz(C)/(n*N);
    NSRE(1)=norm((Y-D*C'),'fro')/reg;
    Dchange(1)=norm((D-D_old),'fro');
    Cchange(1)=norm((C-C_old),'fro')/reg;

%+ Show progress
fprintf(['Running SOUP-DILL',num2str(method),' \n']);
for t = 1:K
    D_old=D;
    C_old=C;
    %+ Show progress
    fprintf('Iteration %2g \n',t);
    tic
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
        bt = mtimes(Y',D(:,j)) - mtimes(C,mtimes(D',D(:,j))) + C(:,j);
        if method ~= 0
%         L1 Norm
            cjt = max(abs(bt)-0.5*(lambda^2),0) .* sign(bt);
        else
%         L0 Norm
            cjt = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));
        end

        C(:,j) = cjt; 
        
        
        %+ Show progress
%         if any(j == floor(J/20:J/20:J)), fprintf('.'); end;
    end
    %+ Show progress
%     fprintf('\n');
    
    if method == 0
        Objfun(t+1)=norm((Y-D*C'),'fro')^2+(lambda^2*nnz(C)); %L0
    else
        Objfun(t+1)=norm((Y-D*C'),'fro')^2+((lambda^2)*sum(sum(abs(C))));  %L1
    end
    Sparsity(t+1)=nnz(C)/(n*N);
    NSRE(t+1)=norm((Y-D*C'),'fro')/reg;
    Dchange(t+1)=norm((D-D_old),'fro');
    Cchange(t+1)=norm((C-C_old),'fro')/reg;
    taxis(t+1)=taxis(t)+toc;
%     DiffCZs(t)=norm(C-x','fro');
        
%     %+ Tracking iterates
%     Dt{t+1} = D; Ct{t+1} = C;
end
    Objfun=gather(Objfun);    
    Sparsity=gather(Sparsity);
    NSRE=gather(NSRE);
    Dchange=gather(Dchange);
    Cchange=gather(Cchange);
    taxis=gather(taxis);
    D=gather(D);
    C=gather(C);
end