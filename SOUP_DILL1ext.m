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

function [D,C,Objfun,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILL1ext(Y,J,lambda,K,L)

global DiffCZs;
global x;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambda = @(b) b.*(abs(b) >= (lambda));
% Dt=0;Ct=0;

% Initial Estimates:
D = genODCT(n,J);
C = zeros(N,J);
Objfun=zeros(1,K);
Sparsity=zeros(1,K);
NSRE=zeros(1,K);
Dchange=zeros(1,K);
Cchange=zeros(1,K);

% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
%+ Show progress
fprintf('Running SOUP-DILLO\n');
for t = 1:K
    D_old=D;
    C_old=C;
    %+ Show progress
    fprintf('Iteration %2g ',t);
    
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
        bt = Y'*D(:,j) - C*(D'*D(:,j)) + C(:,j);
        cjt = max(abs(bt)-0.5*(lambda^2),0) .* sign(bt);
        
        % 3) Dictionary atom update:
        ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
        if any(cjt)
            djt = ht/norm(ht,2);
        else
            djt = v1;
        end
        
        C(:,j) = cjt; 
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
%     DiffCZs(t)=norm(C-x','fro');
        
%     %+ Tracking iterates
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