
function [D,Z,Objfun,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILLO_Npar3(Y,J,lambda,K,L)

global DiffCZs;
global x;
DiffCZs = zeros(1,K);
% Useful definitions:
lambda1=25;
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambda = @(b) b.*(abs(b) >= (sqrt(2*lambda/J)));
H_lambda1 = @(b) b.*(abs(b) >= (lambda1));
% Dt=0;Ct=0;

% Initial Estimates:
D = genODCT(n,J);
Z = zeros(J,N);
Objfun=zeros(1,N+1);
Sparsity=zeros(1,N+1);
NSRE=zeros(1,N+1);
Dchange=zeros(1,N+1);
Cchange=zeros(1,N+1);
% H=Y';
% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
%+ Show progress
fprintf('Running New DL-L0 algorithm\n');
for i = 1:N
    D_old=D;
    C_old=Z;
    %+ Show progress
    fprintf('column %2g  ',i);
    
    for t = 1:K
        % 2) Sparse coding:
        bt = ((eye(J)-(D'*D/J))*Z(:,i)) + D'*Y(:,i)/J;

        
%         L1 Norm
%         Z(:,i) = sign(bt).*max(abs(bt)-(lambda/J),0);


%         L0 Norm
        Z(:,i) = sign(bt).*min(abs(H_lambda(bt)),L);
        % 3) Dictionary atom update:
        
        
        if ( mod(t,10) == 1), fprintf('.'); end;
        
        
    end
        fprintf('\n');
     
%     gi= pinv(Z')*H(:,i);
%     D(i,:)=gi';
        
        %+ Show progress
        
    Objfun(i)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z));
%     Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
    Sparsity(i)=nnz(Z)/(n*N);
    NSRE(i)=norm((Y-D*Z),'fro')/reg;
    Dchange(i)=norm((D-D_old),'fro');
    Cchange(i)=norm((Z-C_old),'fro')/reg;
        
end

   C=Z';
%         Ejpart=(Y-D*C')/J;
for k = 1:200
    for j=1:J
%         ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
        % 2) Sparse coding:
        bt = Y'*D(:,j) - C*(D'*D(:,j)) + C(:,j);
        cjt = min(abs(H_lambda1(bt)),L) .* sign(H_lambda1(bt));
        
        % 3) Dictionary atom update:
        ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
        if any(cjt)
            djt = ht/norm(ht,2);
        else
            djt = v1;
        end
        
        D(:,j) = djt;
    end
end
    fprintf('\n');
    
        
    Objfun(i+1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z));
%     Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
    Sparsity(i+1)=nnz(Z)/(n*N);
    NSRE(i+1)=norm((Y-D*Z),'fro')/reg;
    Dchange(i+1)=norm((D-D_old),'fro');
    Cchange(i+1)=norm((Z-C_old),'fro')/reg;
%+ Show progress
%     fprintf('\n');
    
%     Objfun(i+1)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
%     Sparsity(i+1)=nnz(Z)/(n*N);
%     NSRE(i+1)=norm((Y-D*Z),'fro')/reg;
%     Dchange(i+1)=norm((D-D_old),'fro');
%     Cchange(i+1)=norm((Z-C_old),'fro')/reg;
%     DiffCZs(t)=norm(C-x','fro');
        
%     %+ Tracking iterates
%     Dt{t+1} = D; Ct{t+1} = C;
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