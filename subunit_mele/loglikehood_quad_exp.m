function [f, df, ddf, C, b, a] = loglikehood_quad_exp(V, x, opt, sta, STC, nSpikes)
% loglikehood with quad subunit nonlinearity and exp output nonlinearity
mu = sta';
Lambda = STC;

%% unpack variables
nkt = opt.nkt;
nDim = opt.nDim;
dws = opt.nDim_ws;
dks = opt.nDim_ks;
nModel = opt.nModel;

C = zeros(nDim,nDim);
b =  zeros(nDim,1);
start_id = 1;
for mm = 1:nModel
    nDim_k = dks(mm);
    nDim_w = dws(mm);
    end_id = start_id+nDim_k+nDim_w-1;
    var = V(start_id:end_id);
    w = var(1:nDim_w);
    k = var(1+nDim_w:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    
    K = genToeplitz(k, opt_mm);
    KW = bsxfun(@times, K', w');
    KWK = KW*K;
    C = C+KWK;
    Kw = K'*w;
    b = b+Kw;
    start_id = end_id+1;
end

%% objective
xCx = sum((x*C).*x,2)/2;
tmp0 = xCx+x*b;
max_tmp = max(tmp0);
tmp = tmp0-max_tmp;
expterm = exp(tmp);
Xe = bsxfun(@times, expterm', x');
XX = Xe*x;
sum_expterm = sum(expterm);

f = trace(C*(Lambda+mu*mu'))/2+b'*mu-log(sum_expterm)-max_tmp+log(nSpikes);
f = -f;

%% dic
B = fft(eye(nDim));
BLambdamuB = B*(Lambda+mu*mu')*B';
Bmu = B*mu;
Bmutt = transpose(Bmu');
Bx = B*x';
BxexB = B*XX*B';
Bxe = Bx*expterm;
inv_sum_expterm = 1/(sum_expterm);
inv_sum_expterm2 = inv_sum_expterm^2;
nDim2 = nDim^2;

%% gradient and hessian
df = zeros(size(V));
ddf = zeros(size(V,1),size(V,1));
ddf1 = zeros(size(V,1),size(V,1));
start_id = 1;
mats = cell(nModel,1);

for mm = 1:nModel
    
    nDim_k = dks(mm);
    nDim_w = dws(mm);
    end_id = start_id+nDim_k+nDim_w-1;
    var = V(start_id:end_id);
    mat.w = var(1:nDim_w);
    mat.k = var(1+nDim_w:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    
    mat.K = genToeplitz(mat.k, opt_mm);
    Kmu = mat.K*mu;
    wid = nDim_k:nkt:nDim;
    Bw = B(:,wid);
    Bk = B(:,1:nDim_k);
    BWB = bsxfun(@times, Bw, mat.w')*Bw';
    BK = Bk*mat.k;
    BW = Bw*mat.w;
    BKt = Bk';
    BwdBK = bsxfun(@times, Bw', transpose(BK));
    
    %% gradient
    %% nDim_w
    mat.Kx = x*mat.K';
    mat.Kx2 = mat.Kx.^2/2+mat.Kx;
    mat.eKx2 = bsxfun(@times, expterm, mat.Kx2);
    mat.dex = sum(mat.eKx2, 1)';
    
    g1 = diag(mat.K*Lambda*mat.K')/2;
    g2 = (Kmu+1).^2/2;
    g3 = inv_sum_expterm*mat.dex+0.5;
    df_w = g1+g2-g3;
    
    %% nDim_k
    % traceterm = trace(BLambdamuPhiB*diag(BK)'*BWBdBK)/nDim2/2
    % K0 = BwdBK*B/nDim;
    
    A = bsxfun(@times, BLambdamuB, BK')*BWB;
    g1 = BKt*transpose(diag(A))'/nDim2;
    g2 = BKt*(Bmutt.*BW)/nDim;
    
    AMk = transpose(bsxfun(@times, Bx', BK')*BWB).*Bx/nDim2;
    AMw = bsxfun(@times, transpose(BW'), Bx)/nDim;
    AM = AMk+AMw;
    mat.alpha = real(Bk'*transpose(AM*expterm)');
    g3 = inv_sum_expterm*mat.alpha;
    
    df_k = real(g1+g2-g3);
    
    df(start_id:end_id) = -real([df_w; df_k]);
    
    %% ww
    H1 = bsxfun(@times, inv_sum_expterm2*mat.dex', mat.dex);
    H2 = -inv_sum_expterm*(mat.Kx2)'*mat.eKx2;
    ddf_ww = H1+H2;
    
    %% kk
    H1 = real(BKt*(transpose(BLambdamuB).*BWB)*Bk/nDim2);
    H2 = bsxfun(@times, inv_sum_expterm2*mat.alpha', mat.alpha);
    H3 = -real(inv_sum_expterm*Bk'*(BWB.*transpose(BxexB))*Bk/nDim2);
    
    alpha_K = Bk'*transpose(AMk)';
    alpha_w = Bk'*transpose(AMw)';
    mat.alpha_Kw = alpha_K+alpha_w;
    mat.alpha_Kx_exp = bsxfun(@times, mat.alpha_Kw, expterm');
    H4 = real(-inv_sum_expterm*mat.alpha_Kx_exp*transpose(mat.alpha_Kw));
    
    ddf_kk = real(H1+H2+H3+H4);
    
    %% kw
    A = BwdBK*BLambdamuB;
    M = Bw;
    H1 = real(BKt*(transpose(A).*M)/nDim2);
    H2 = real(BKt*bsxfun(@times, Bmutt, Bw)/nDim);
    H3 = bsxfun(@times, inv_sum_expterm2*mat.dex', mat.alpha);
    
    H41 = -inv_sum_expterm*real(Bk'*(Bw.*transpose(BwdBK*BxexB)))/nDim2;
    H42 = -inv_sum_expterm*real(bsxfun(@times, Bk', Bxe')*Bw)/nDim;
    H4 = H41+H42;
    
    H5 = -inv_sum_expterm*mat.alpha_Kx_exp*mat.Kx2;
    
    ddf_kw = real(H1+H2+H3+H4+H5);
    
    ddf(start_id:end_id, start_id:end_id) = [ddf_ww ddf_kw'; ddf_kw ddf_kk];
    start_id = end_id+1;
    mats{mm} = mat;
end


%% hessian off diag
start_id = 1;
for mm = 1:nModel
    
    nDim_k = dks(mm);
    nDim_w = dws(mm);
    end_id = start_id+nDim_k+nDim_w-1;
    
    mat = mats{mm};
    
    start_id1 = end_id+1;
    for nn = mm+1:nModel
        nDim_k1 = dks(nn);
        dw1 = dws(nn);
        end_id1 = start_id1+nDim_k1+dw1-1;
        
        mat1 = mats{nn};
        
        %% ww1
        H1 = bsxfun(@times, inv_sum_expterm2*mat1.dex', mat.dex);
        H2 = -inv_sum_expterm*(mat.Kx2)'*mat1.eKx2;
        ddf_ww1 = H1+H2;
        
        %% kk1
        H2 = bsxfun(@times, inv_sum_expterm2*mat1.alpha', mat.alpha);
        H4 = real(-inv_sum_expterm*mat1.alpha_Kx_exp*transpose(mat.alpha_Kw))';
        ddf_kk1 = H2+H4;
        
        %% kw1
        H3 = bsxfun(@times, inv_sum_expterm2*mat1.dex', mat.alpha);
        H5 = -inv_sum_expterm*mat.alpha_Kx_exp*mat1.Kx2;
        ddf_kw1 = H3+H5;
        
        %% wk1
        H3 = bsxfun(@times, 1/(sum_expterm)^2*mat.dex', mat1.alpha);
        H5 = -1/(sum_expterm)*mat1.alpha_Kx_exp*mat.Kx2;
        ddf_wk1 = (H3+H5)';
        
        ddf1(start_id:end_id, start_id1:end_id1) = [ddf_ww1 ddf_wk1; ddf_kw1 ddf_kk1];
        
        start_id1 = end_id1+1;
    end
    start_id = end_id+1;
    
end

ddf = -real(ddf+ddf1+ddf1');

%% a_log
a = log(opt.nSpikes)-log(sum_expterm)-max_tmp;

%% smoothing
if ~(~isfield(opt, 'smoothing') || opt.smoothing == 0)
    invKse = opt.invKse*opt.lambda_w;
    if opt.smoothk
        invKse1 = opt.invKse1*opt.lambda_k;
    end
    
    start_id = 1;
    for mm = 1:nModel
        nDim_w = dws(mm);
        nDim_k = dks(mm);
        
        end_id = start_id+nDim_k+nDim_w-1;
        var = V(start_id:end_id);
        w = var(1:nDim_w);
        if opt.smoothing == 2
            w = w/norm(w);
        end
        f = f + w'*invKse*w/2;
        df(start_id:start_id+nDim_w-1) = df(start_id:start_id+nDim_w-1) + invKse*w;
        ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) = ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) + invKse;
        
        if opt.smoothk
            k = var(1+nDim_w:end);
            f = f + k'*invKse1*k/2;
            df(start_id+nDim_w:end_id) = df(start_id+nDim_w:end_id) + invKse1*k;
            ddf(start_id+nDim_w:end_id, start_id+nDim_w:end_id) = ddf(start_id+nDim_w:end_id, start_id+nDim_w:end_id) + invKse1;
        end
        
        start_id = end_id+1;
    end
end
%%
clearvars -except f df ddf C b a
