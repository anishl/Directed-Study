
function [D,Z,Objfun,Sparsity,NSRE,Dchange,Cchange,taxis] = SOUP_DILLO_Npar2(Y,J,lambda,K,L,alpha)

global DiffCZs;
global x;
global Z_start;
global alt;
global method;
global DL;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = gpuArray(In(:,1));
% alpha=1.0;
% Dt=0;Ct=0;
% niter=600;
% Initial Estimates:
% D = genODCT(n,J);
D=gpuArray(x);
% Z = gpuArray(zeros(J,N));
Z=gpuArray(Z_start);
Objfun=gpuArray(zeros(1,2*K+1));
Sparsity=gpuArray(zeros(1,2*K+1));
NSRE=gpuArray(zeros(1,2*K+1));
Dchange=gpuArray(zeros(1,2*K+1));
Cchange=gpuArray(zeros(1,2*K+1));
taxis=gpuArray(zeros(1,2*K+1));
Y=gpuArray(Y);
D_old=D;
C_old=Z;
V1=repmat(v1,1,size(D,2));
% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
%+ Show progress
fprintf(['Running New DL-L',num2str(method),' algorithm\n']);

    if method == 0
        Objfun(1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z)); %L0
    else
        Objfun(1)=norm((Y-D*Z),'fro')^2+((lambda^2)*sum(sum(abs(Z))));    %L1
    end
        Sparsity(1)=nnz(Z)/(n*N);
        NSRE(1)=norm((Y-D*Z),'fro')/reg;
        Dchange(1)=norm((D-D_old),'fro');
        Cchange(1)=norm((Z-C_old),'fro')/reg;

% lam = alpha*max(eig(mtimes(D',D)))+(1-alpha)*J;
% lam=J;

for t = 1:K
    D_old=D;
    C_old=Z';
    %+ Show progress
    
    lam = alpha*max(eig(mtimes(D',D)))+(1-alpha)*J;
    H_lambda = @(b) b.*(abs(b) >= (lambda/sqrt(lam)));
    DtY = mtimes(D',Y);
%     if(t==1)
        fprintf('iteration %2g \n Z update',t);
        tic;

        M = (gpuArray(eye(J))-mtimes(D',D/lam));
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
        Objfun(2*t)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z)); %L0
    else
        Objfun(2*t)=norm((Y-D*Z),'fro')^2+((lambda^2)*sum(sum(abs(Z)))); %L1
    end
        Sparsity(2*t)=nnz(Z)/(n*N);
        NSRE(2*t)=norm((Y-D*Z),'fro')/reg;
        Dchange(2*t)=norm((D-D_old),'fro');
        Cchange(2*t)=norm((Z'-C_old),'fro')/reg;
        taxis(2*t)=taxis(2*t-1)+toc;

        fprintf('\n');
        
% % % % % % % % % % % % % % % %D update start % % % % % % % % % % % % % % %
        tic;
        if DL == 1
         if mod(t,alt)==0
            C=Z';
% % % % % % %             SOUP-ORIGINAL
%             
%         for niter =1:3
            for j=1:J
                ht = Y*C(:,j) - D*(C'*C(:,j)) + D(:,j)*dot(C(:,j),C(:,j));
                if any(C(:,j))
                    djt = ht/norm(ht,2);
                else
                    djt = v1;
                end
                D(:,j)=djt;
            end
%         end
% 
% % % % % % % %             SOUP-ORIGINAL
            

% % % % % % % %             J-Par D-Update
% 
     
%             ht = Y*cjt - D*(C'*cjt) + D(:,j)*dot(C(:,j),cjt);
%             ht=Ejpart*C(:,j)+(D(:,j)*(norm(C(:,j))^2));

%%%%%% Parallel Code Starts Here

%               Ejpart=(Y-mtimes(D,C'))/J;
%               Ht = mtimes(Ejpart,C)+ mtimes(D,diag(diag(mtimes(C',C))));
%               D = mtimes(mtimes(Ht,sqrt(pinv(diag(diag(mtimes(Ht',Ht)))))),double(diag(any(C)))) ...
%                 + mtimes(V1,double(diag(1-any(C))));

%%%%%% Parallel Code Ends Here
            
%             if any(C(:,j))
%                 djt = ht/norm(ht,2);
%             else
%                 djt = v1;
%             end
% 
%              D(:,j) = djt;
        
% 
% % % % % % % %             J-Par D-Update
         end
        end
        
% % % % % % % % % % % % % % % % %D update end % % % % % % % % % % % % % % %
        
        
    if method == 0
        Objfun(2*t+1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z)); %L0
    else
        Objfun(2*t+1)=norm((Y-D*Z),'fro')^2+((lambda^2)*sum(sum(abs(Z))));    %L1
    end
        Sparsity(2*t+1)=nnz(Z)/(n*N);
        NSRE(2*t+1)=norm((Y-D*Z),'fro')/reg;
        Dchange(2*t+1)=norm((D-D_old),'fro');
        Cchange(2*t+1)=norm((Z'-C_old),'fro')/reg;
        taxis(2*t+1)=taxis(2*t)+toc;
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