
function [D,Z,Objfun,Sparsity,NSRE,Dchange,Cchange,taxis] = SOUP_DILLO_Npar_seq(Y,J,lambda,K,L,alpha)

global DiffCZs;
global x;
global Z_start;
global alt;
DiffCZs = zeros(1,K);
% Useful definitions:
[n,N] = size(Y);
In = eye(n); v1 = In(:,1);
% alpha=1.0;
% Dt=0;Ct=0;
% niter=600;
% Initial Estimates:
% D = genODCT(n,J);
D=gpuArray(x);
% Z = gpuArray(zeros(J,N));
Z=gpuArray(Z_start);
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
alpha=1.0;
% %+ Tracking iterates
% Dt = cell(1,K+1); Ct = cell(1,K+1);
% Dt{1} = D; Ct{1} = C;
reg=norm(Y,'fro');
%+ Show progress
fprintf('Running New DL-L0 algorithm\n');


        Objfun(1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z));
%         Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
        Sparsity(1)=nnz(Z)/(n*N);
        NSRE(1)=norm((Y-D*Z),'fro')/reg;
        Dchange(1)=norm((D-D_old),'fro');
        Cchange(1)=norm((Z-C_old),'fro')/reg;

% lam = alpha*max(eig(mtimes(D',D)))+(1-alpha)*J;
% lam=J;

for t = 1:K
    D_old=D;
    C_old=Z';
    fprintf('Iteration: %d\n',t);
 
        
% % % % % % % % % % % % % % % %D update start % % % % % % % % % % % % % % %
        tic;
% % % % % % %             SOUP-ORIGINAL
%             
%         for niter =1:3
            for j=1:J
                for iter=1:2
                    lam = alpha*max(eig(mtimes(D',D)))+(1-alpha)*J;
                    H_lambda = @(b) b.*(abs(b) >= (lambda/sqrt(lam)));
                    DtY = mtimes(D',Y);

                    M = (gpuArray(eye(J))-mtimes(D',D/lam));
                    Bt=mtimes(M,Z)+DtY/lam;
                    Z = sign(Bt).*min(abs(H_lambda(Bt)),L);
                end
                
                C=Z';
                
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
            
%         end
        
% % % % % % % % % % % % % % % % %D update end % % % % % % % % % % % % % % %
        
        

        Objfun(t+1)=norm((Y-D*Z),'fro')^2+((lambda^2)*nnz(Z));
%         Objfun(i)=norm((Y-D*Z),'fro')^2+(lambda*sum(sum(abs(Z))));
        Sparsity(t+1)=nnz(Z)/(n*N);
        NSRE(t+1)=norm((Y-D*Z),'fro')/reg;
        Dchange(t+1)=norm((D-D_old),'fro');
        Cchange(t+1)=norm((Z'-C_old),'fro')/reg;
        taxis(t+1)=taxis(t)+toc;
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