function M1 = fillupcirc(M)
M(abs(M)<1e-10) = 0;

Ml = tril(M);
Ml1 = fliptriul(flipud(Ml));
Mu = triu(M);
Mu1 = flipud(fliptriur(Mu));
Mu1 = [zeros(1,size(Mu1,2)); Mu1(1:end-1,:)];

M1 = Mu1+Ml1;