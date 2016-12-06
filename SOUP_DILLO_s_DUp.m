
function [D,C,Objfun,Sparsity,NSRE,Dchange,Cchange,taxis] = SOUP_DILLO_s_DUp(Y,J,lambda,K,L)

global DiffCZs;
global x;
global y;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambda = @(b) b.*(abs(b) >= (lambda));
% Dt=0;Ct=0;

% Initial Estimates:
D = gpuArray(x);
% C = gpuArray(zeros(N,J));
% C = gpuArray(x');
C=gpuArray(y);
Objfun=gpuArray(zeros(1,J*K+1));
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

    Objfun(1)=norm((Y-D*C'),'fro')^2+((lambda^2)*nnz(C));
    Sparsity(1)=nnz(C)/(n*N);
    NSRE(1)=norm((Y-D*C'),'fro')/reg;
    Dchange(1)=norm((D-D_old),'fro');
    Cchange(1)=norm((C-C_old),'fro')/reg;

%+ Show progress
fprintf('Running SOUP-DILLO\n');
for t = 1:K
    D_old=D;
    C_old=C;
    %+ Show progress
    fprintf('Iteration %2g ',t);
    tic
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
%         bt = mtimes(Y',D(:,j)) - mtimes(C,mtimes(D',D(:,j))) + C(:,j);
%         L1 Norm
%         cjt = max(abs(bt)-0.5*(lambda^2),0) .* sign(bt);

%         L0 Norm
%         cjt = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));
        cjt=C(:,j);
% %         3) Dictionary atom update:
        ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
        if any(cjt)
            djt = ht/norm(ht,2);
        else
            djt = v1;
        end
% %       D update end  
%         C(:,j) = cjt; 
        D(:,j) = djt;% D update
        Objfun(J*(t-1)+j+1)=norm((Y-D*C'),'fro')^2;
        
        %+ Show progress
%         if any(j == floor(J/20:J/20:J)), fprintf('.'); end;
    end
    %+ Show progress
    fprintf('\n');
    
    Objfun(t+1)=norm((Y-D*C'),'fro')^2+(lambda^2*nnz(C));
    Sparsity(t+1)=nnz(C)/(n*N);
    NSRE(t+1)=norm((Y-D*C'),'fro')/reg;
    Dchange(t+1)=norm((D-D_old),'fro');
    Cchange(t+1)=norm((C-C_old),'fro')/reg;
    taxis(t+1)=taxis(t)+toc;
%     DiffCZs(t)=norm(C-x','fro');
        
%     %+ Tracking iterates
%     Dt{t+1} = D; Ct{t+1} = C;
end

end
