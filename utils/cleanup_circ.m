function MM = cleanup_circ(M)

M1 = fillupcirc(M);
mask = M1~=0;
mn = sum(mask,2);
MM = sum(M1,2)./mn;
MM = transpose(MM');