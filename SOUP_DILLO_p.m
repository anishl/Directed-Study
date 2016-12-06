function [D,C,Objfun,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILLO_p(Y,J,lambda,K,L)

global DiffCZp;
global x;
DiffCZp = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambda = @(b) b.*(abs(b) >= (lambda/sqrt(J)));
% Dt=0;Ct=0;

% Initial Estimates:
D = genODCT(n,J);
C = zeros(N,J);
Objfun=zeros(1,K);
Sparsity=zeros(1,K);
NSRE=zeros(1,K);
Dchange=zeros(1,K);
Cchange=zeros(1,K);
%+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
% D_old=D;
% C_old=C;
reg=norm(Y,'fro');
%+ Show progress
fprintf('Running SOUP-DILLO\n');
for t = 1:K
    D_old=D;
    C_old=C;
    %+ Show progress
    fprintf('Iteration %2g ',t);
    Ejpart=(Y-D*C')/J;
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
%         bt = Y'*D(:,j) - C*(D'*D(:,j)) + C(:,j);
%         cjt = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));
        bt = (Ejpart'*D(:,j))+C(:,j);
        C(:,j) = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));        
    end
    
% 3) Dictionary atom update:
    Ejpart=(Y-D*C')/J;
    for j=1:J
%         ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
        ht=Ejpart*C(:,j)+(D(:,j)*(norm(C(:,j))^2));
        if any(C(:,j))
            djt = ht/norm(ht,2);
        else
            djt = v1;
        end
        
         D(:,j) = djt;
        
        %+ Show progress
        if any(j == floor(J/20:J/20:J)), fprintf('.'); end;
    end
    %+ Show progress
    fprintf('\n');
    
    Objfun(t)=norm((Y-D*C'),'fro')^2+(lambda^2*nnz(C));
    Sparsity(t)=nnz(C)/(n*N);
    NSRE(t)=norm((Y-D*C'),'fro')/reg;
    Dchange(t)=norm((D-D_old),'fro');
    Cchange(t)=norm((C-C_old),'fro')/reg;
    
    %for comparison in L1_closedform
    DiffCZp(t)=norm(C-x','fro');
    
    %+ Tracking iterates
%     Dt{t+1} = D; Ct{t+1} = C;
end

end

% Generate 2D Overcomplete DCT
function ODCT = genODCT(n,J)

sqrtn = ceil(sqrt(n)); sqrtJ = ceil(sqrt(J));

ODCT = zeros(sqrtn,sqrtJ);
ODCT(:,1) = 1/sqrt(sqrtn);
for j = 2:sqrtJ
  v = cos(pi*(j-1)/sqrtJ * (0:sqrtn-1))'; v = v-mean(v);
  ODCT(:,j) = v/norm(v);
end
ODCT = kron(ODCT,ODCT);

ODCT = ODCT(1:n,1:J);

end