function [A, invA] = svd_inv(A, thresh)

[U, S, V] = svd(A);
dS = diag(S);
dSid = dS>thresh*max(dS);
uu = U(:,dSid);
vv = V(:,dSid);
ss = S(dSid,dSid);
A = uu*ss*vv';
invA = uu*diag(1./diag(ss))*vv';

