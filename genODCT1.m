function ODCT = genODCT1(n,J, center)

% sqrtn = ceil(sqrt(n)); sqrtJ = ceil(sqrt(J));
% 
% ODCT = zeros(sqrtn,sqrtJ);
% ODCT(:,1) = 1/sqrt(sqrtn);
% for j = 2:sqrtJ
%   v = cos(pi*(j-1)/sqrtJ * (0:sqrtn-1))'; v = v-mean(v);
%   ODCT(:,j) = v/norm(v);
% end

ODCT = idct(eye(sqrt(J)));
ODCT = ODCT(1:sqrt(n),:);
% ODCT = (sqrt(n))*kron(ODCT,ODCT);
ODCT = kron(ODCT,ODCT);
%%
if center == 1
    ODCT=[ODCT(:,1) ODCT(:,2:end)-repmat(mean(ODCT(:,2:end)),n,1)];
end
%%
for i=1:J
   ODCT(:,i)=ODCT(:,i)/norm(ODCT(:,i)); 
end
% ODCT = ODCT(1:n,1:n);

end