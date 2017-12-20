function [C1, K] = genCK(w, k, opt)
%% use Toeplitz matrix with no circulation
C1 = zeros(opt.nDim);
K = zeros(opt.nDim_w, opt.nDim);

for i=1:size(k,2)
    K = genToeplitz(k(:,i), opt);
    C1 = C1+K'*diag(w(:,i))*K;
end
