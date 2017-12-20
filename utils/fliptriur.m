function A1 = fliptriur(A)
% a b c d     a f k q
% 0 f g h --> 0 b g d
% 0 0 k l     0 0 c h
% 0 0 0 q     0 0 0 d
nn = size(A,1);
[ri,ci] = triuinds(nn);
id = (ci-1)*nn+ri;
dif = nn+1;
ff = 1:-1:(3-dif);
mm = repmat(ff',1,nn)+repmat(dif*(0:nn-1),nn,1);
mvec = mm(id);
M = unvecUpperMtxFromTriu(mvec)';
Mvec = M(:);
zz = Mvec==0;
nzz = ~zz;
Mvec(zz) = [];

Avec = A(Mvec);
A1 = zeros(nn^2,1);
A1(nzz) = Avec;
A1 = reshape(A1,nn,nn)';
