function K = genToeplitz(k, opt)
n = opt.nDim;
d = opt.nDim_k;
nkt = opt.nkt;
% k = flipud(k);
kk = [k; zeros(n-d,1)];
K = fliplr(makeStimRows(kk,n)); % Toeplitz matrix
K = K(d:end,:);

ind = 1:size(K,1);
if nkt ~= 1
    ind = mod(ind, nkt)==1;
end
K = K(ind,:);