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