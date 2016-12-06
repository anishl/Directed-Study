function [X,Objfun,sprs,NSRE] = DILL1_test(Y,D,lambda)
[n,N]=size(Y);
[~,J]=size(D);
X=zeros(J,N);
for i=1:N
    z=solveLasso(Y(:,i),D,lambda);
    X(:,i)=z.beta;
     if any(i == floor(N/20:N/20:N)), fprintf('.'); end;
end
Objfun=norm(Y-(D*X),'fro')^2+(lambda*sum(sum(abs(X))));
sprs=nnz(X)/(n*N);
NSRE=norm(Y-(D*X),'fro')/norm(Y,'fro');
end
