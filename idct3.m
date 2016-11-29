function [Yd]= idct3(Y)
[nx,ny,np]=size(Y);
if (np==1)
    Yd=dct2(Y);
else
    Yd=permute(reshape(idct(reshape(permute(reshape(idct(reshape(Y,nx,[])),nx,ny,np),[2,1,3]),ny,[])),ny,nx,np),[2 1 3]);
end

