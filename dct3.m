function [Yd]= dct3(Y)
[nx,ny,np]=size(Y);
% if (np==1)
%     Yd=dct2(Y);
% else
    Yd=permute(reshape(dct(reshape(permute(reshape(dct(reshape(Y,nx,[])),nx,ny,np),[2,1,3]),ny,[])),ny,nx,np),[2 1 3]);
% end

