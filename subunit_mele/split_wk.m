function [var_new1, w_new, k_new] = split_wk(var_new, opt)

var_new1 = reshape(var_new, length(var_new)/opt.nModel, []);
w_new = var_new1(1:opt.nDim_ws(1),:);
k_new = var_new1(1+opt.nDim_ws(1):end,:);
% k_new_norm = sqrt(sum(k_new.^2,1));
% w_new = bsxfun(@times,w_new, k_new_norm.^2);
% k_new = bsxfun(@times,k_new, 1./k_new_norm);
var_new1 = [w_new; k_new];
var_new1 = var_new1(:);
