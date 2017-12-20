function [KWKs, Kws] = from_var_to_C(V, opt)

%% unpack variables
nkt = opt.nkt;
n = opt.nDim;
dws = opt.nDim_ws;
dks = opt.nDim_ks;
nModel = opt.nModel;

KWKs = zeros(n,n);
Kws =  zeros(n,1);
start_id = 1;
for mm = 1:nModel
    dk = dks(mm);
    dw = dws(mm);
    end_id = start_id+dk+dw-1;
    var = V(start_id:end_id);
    w = var(1:dw);
    k = var(1+dw:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    
    K = genToeplitz(k, opt_mm);
    KW = bsxfun(@times, K', w');
    KWK = KW*K;
    KWKs = KWKs+KWK;
    Kw = K'*w;
    Kws = Kws+Kw;
    start_id = end_id+1;
end