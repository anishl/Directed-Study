function [D,C,Dt,Ct] = SOUP_DILLO(Y,J,lambda,K,L)

% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
H_lambda = @(b) b.*(abs(b) >= lambda);

% Initial Estimates:
D = genODCT(n,J);
C = zeros(N,J);

%+ Tracking iterates
Dt = cell(1,K+1); Ct = cell(1,K+1);
Dt{1} = D; Ct{1} = C;

%+ Show progress
fprintf('Running SOUP-DILLO\n');
for t = 1:K
    %+ Show progress
    fprintf('Iteration %2g ',t);
    
    for j = 1:J
        % 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
        %    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]
        
        % 2) Sparse coding:
        bt = Y'*D(:,j) - C*(D'*D(:,j)) + C(:,j);
        cjt = min(abs(H_lambda(bt)),L) .* sign(H_lambda(bt));
        
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
    
    %+ Tracking iterates
    Dt{t+1} = D; Ct{t+1} = C;
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