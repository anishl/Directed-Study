% % SOUP_DILLO  Learn dictionary using SOUP-DILLO (SOUP-DIL with l0 "norm").
% %   [D,C,Dt,Ct] = SOUP_DILLO(Y,J,lambda,K,L) learns a dictionary D of J
% %   atoms with corresponding coefficients C for the data matrix Y by
% %   running K iterations of SOUP-DILLO with the tuning parameter lambda.
% %   Dt and Ct are cell arrays containing all the iterates.
% %   
% %   The algorithm attempts to solve the minimization:
% %
% %                     2         2
% %     min  ||Y - DC'||  + lambda  ||C||
% %     D,C             F                0
% %
% %     s.t. ||d ||  = 1, ||c ||    <= L   for all j
% %             j  2         j  inf
% %
% %   The algorithm can be found in the arXiv paper: 1511.06333v3
% %     S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, "Efficient Sum
% %     of Sparse Outer Products Dictionary Learning (SOUP-DIL)," 2016.
% 
% %   Author: David Hong (dahong@umich.edu)
% %   Last edited: 1 June 2016

function [D,C,Objfun,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILLO_pCUp(Y,J,lambda,K,L)

global DiffCZp;
global x;
DiffCZp = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambdaJ = @(b) b.*(abs(b) >= (lambda/sqrt(J)));
% Dt=0;Ct=0;

% Initial Estimates:
% D = gpuArray(genODCT(n,J));
D=gpuArray(x);
C = gpuArray(zeros(N,J));
% C=gpuArray(x');
Objfun=gpuArray(zeros(1,K+1));
Sparsity=gpuArray(zeros(1,K+1));
NSRE=gpuArray(zeros(1,K+1));
Dchange=gpuArray(zeros(1,K+1));
Cchange=gpuArray(zeros(1,K+1));
Y=gpuArray(Y);

D_old=D;
C_old=C;


%+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
% D_old=D;
% C_old=C;
reg=norm(Y,'fro');
%+ Show progress

        Objfun(1)=norm((Y-D*C'),'fro')^2+((lambda^2)*nnz(C));
%         Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
        Sparsity(1)=nnz(C)/(n*N);
        NSRE(1)=norm((Y-D*C'),'fro')/reg;
        Dchange(1)=norm((D-D_old),'fro');
        Cchange(1)=norm((C-C_old),'fro')/reg;



fprintf('Running SOUP-DILLO\n');
for t = 1:K
    D_old=D;
    C_old=C;
    %+ Show progress
    fprintf('Iteration %2g ',t);
    Ejpart=minus(Y,mtimes(D,C'))/J;
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
%         bt = Y'*D(:,j) - C*(D'*D(:,j)) + C(:,j);
%         cjt = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));
        bt = mtimes(Ejpart',D(:,j))+C(:,j);
        C(:,j) = min(abs(H_lambdaJ(bt)),L) .* sign(H_lambdaJ(bt));        
%         C(:,j) = sign(bt).*max(minus(abs(bt),(0.5*(lambda^2)/J)),0);
    end
    
% 3) Dictionary atom update:
%     Ejpart=(Y-D*C')/J;
%     for j=1:J
%         ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
%         ht=Ejpart*C(:,j)+(D(:,j)*(norm(C(:,j))^2));
%         if any(C(:,j))
%             djt = ht/norm(ht,2);
%         else
%             djt = v1;
%         end
%         
%          D(:,j) = djt;
        
        %+ Show progress
        if any(j == floor(J/20:J/20:J)), fprintf('.'); end;
%     end
    %+ Show progress
    fprintf('\n');
    
        Objfun(t+1)=norm((Y-D*C'),'fro')^2+((lambda^2)*nnz(C));
%         Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
        Sparsity(t+1)=nnz(C)/(n*N);
        NSRE(t+1)=norm((Y-D*C'),'fro')/reg;
        Dchange(t+1)=norm((D-D_old),'fro');
        Cchange(t+1)=norm((C-C_old),'fro')/reg;
    
    %for comparison in L1_closedform
%     DiffCZp(t)=norm(C-x','fro');
    
    %+ Tracking iterates
%     Dt{t+1} = D; Ct{t+1} = C;
end

end

% Generate 2D Overcomplete DCT
function ODCT = genODCT(n,J)

% sqrtn = ceil(sqrt(n)); sqrtJ = ceil(sqrt(J));
% 
% ODCT = zeros(sqrtn,sqrtJ);
% ODCT(:,1) = 1/sqrt(sqrtn);
% for j = 2:sqrtJ
%   v = cos(pi*(j-1)/sqrtJ * (0:sqrtn-1))'; v = v-mean(v);
%   ODCT(:,j) = v/norm(v);
% end

ODCT = idct(eye(n));

ODCT = kron(ODCT,ODCT);

ODCT = ODCT(1:n,1:J);

end