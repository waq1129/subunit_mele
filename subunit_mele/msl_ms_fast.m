function [f, df, ddf, KWKs, Kws] = msl_ms_fast(V, C0, b0, opt)
%%%%%%%%%%%%%%%%%%%%%%
% MSL estimator
%%%%%%%%%%%%%%%%%%%%%%
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
%% C_bstc-K'*diag(w)*K
L = C0 - KWKs;

%% b_bstc-K'*w
Lb = b0 - Kws;

%% objective
f = sum(sum(L.^2))+sum(Lb.^2);

%%
df = zeros(size(V));
ddf = zeros(size(V,1),size(V,1));
ddf1 = zeros(size(V,1),size(V,1));
start_id = 1;
B = fft(eye(n));
BLB = B*L*B';
BLb = B*Lb;
mats = cell(nModel,1);
for mm = 1:nModel
    dk = dks(mm);
    dw = dws(mm);
    end_id = start_id+dk+dw-1;
    var = V(start_id:end_id);
    mat.w = var(1:dw);
    mat.k = var(1+dw:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    
    % dictionary
    
    mat.K = genToeplitz(mat.k, opt_mm);
    mat.KW = bsxfun(@times, mat.K', mat.w');
    wid = dk:nkt:n;
    mat.Bk = B(:,1:dk);
    mat.Bw = B(:,wid);
    mat.BK = mat.Bk*mat.k;
    mat.BW = mat.Bw*mat.w;
    mat.Bwdw = bsxfun(@times, mat.Bw, mat.w');
    mat.Bktt = transpose(mat.Bk)';
    mat.BWB = mat.Bwdw*mat.Bw';
    mat.BwdBK = bsxfun(@times, mat.Bw', transpose(mat.BK));
    mat.BWBdBK = bsxfun(@times, mat.BWB, transpose(mat.BK));
    mat.BwdwBwdBK = mat.Bwdw*mat.BwdBK;
    mat.BkdBW = bsxfun(@times, mat.Bk', transpose(mat.BW));
    
    BWBdBKBwdBK = mat.BWBdBK*mat.BwdBK';
    BwdBKBLB = mat.BwdBK*BLB;
    
    %% gradient
    
    % d_w
    df_w = -2*diag(BwdBKBLB*mat.BwdBK')/n^2-2*mat.BwdBK*BLb/n;
    
    % d_k
    A = mat.BWBdBK*BLB;
    g1 = real(mat.Bk'*diag(A)/n^2);
    
    g2 = real(mat.Bk'*(transpose(BLb').*mat.BW)/n);
    df_k = -4*g1-2*g2;
    
    % assemble df
    df(start_id:end_id) = [df_w; df_k];
    
    %% hessian
    % ww
    KK = mat.K*mat.K';
    ddf_ww = 2*KK.^2+2*KK;
    
    % kk
    
    % derivative of LK'diag(mat.w)
    A = BLB;
    M = mat.BWB;
    H1 = real(mat.Bk'*(transpose(A).*M)*mat.Bk/n^2);
    
    % derivative of mat.K'A, A=diag(mat.w)KK'diag(mat.w)
    M = BWBdBKBwdBK*mat.Bwdw';
    H2 = real(bsxfun(@times, mat.Bk', transpose(diag(M)))*mat.Bk/n^2);
    
    % derivative of AKA, A=mat.K'diag(mat.w)
    A = mat.BwdwBwdBK;
    M = mat.BWBdBK;
    H3 = real(mat.Bk'*(transpose(A).*M)*mat.Bktt/n^2);
    
    %
    H4 = real(mat.BkdBW*mat.BkdBW'/n);
    
    ddf_kk = -4*(H1-H2-H3)+2*H4;
    
    % kw
    A = BwdBKBLB;
    M = mat.Bw;
    H1 = real(mat.Bk'*(transpose(A).*M)/n^2);
    
    A = mat.BwdBK;
    M = BWBdBKBwdBK;
    H2 = real(mat.Bk'*(transpose(A).*M)/n^2);
    
    H3 = real(bsxfun(@times, mat.Bk', BLb')*mat.Bw/n);
    
    H4 = real(transpose(mat.BkdBW')*mat.BwdBK'/n);
    
    ddf_kw = -4*(H1-H2)-2*(H3-H4);
    
    ddf(start_id:end_id, start_id:end_id) = [ddf_ww ddf_kw'; ddf_kw ddf_kk];
    
    mats{mm} = mat;
    start_id = end_id+1;
end

start_id = 1;
for mm = 1:nModel
    dk = dks(mm);
    dw = dws(mm);
    end_id = start_id+dk+dw-1;
    mat = mats{mm};
    Bk = mat.Bk;
    Bktt = mat.Bktt;
    
    %%
    start_id1 = end_id+1;
    
    for nn = mm+1:nModel
        dk1 = dks(nn);
        dw1 = dws(nn);
        end_id1 = start_id1+dk1+dw1-1;
        mat1 = mats{nn};
        
        % dictionary
        BWBdBKBwdBK1 = mat.BWBdBK*mat1.BwdBK';
        BW1BdBK1BwdBK = mat1.BWBdBK*mat.BwdBK';
        
        % ww
        KK1 = mat.K*mat1.K';
        ddf_ww1 = 2*KK1.^2+2*KK1;
        
        % kk
        % derivative of mat.K'A, A=diag(w)KK'diag(w)
        M = BWBdBKBwdBK1*mat1.Bwdw';
        H2 = real(bsxfun(@times, Bk', transpose(diag(M)))*Bk/n^2);
        
        % derivative of AKA, A=mat.K'diag(w)
        A = mat1.BwdwBwdBK;
        M = mat.BWBdBK;
        H3 = real(Bk'*(transpose(A).*M)*Bktt/n^2);
        
        %
        H4 = real(mat.BkdBW*mat1.BkdBW'/n);
        
        ddf_kk1 = real(4*(H2+H3)+2*H4);
        
        % kw1
        A = mat1.BwdBK;
        M = BWBdBKBwdBK1;
        H2 = real(Bk'*(transpose(A).*M)/n^2);
        
        H4 = real(transpose(mat.BkdBW')*mat1.BwdBK'/n);
        
        ddf_kw1 = real(4*H2+2*H4);
        
        % wk1
        A = mat.BwdBK;
        M = BW1BdBK1BwdBK;
        H2 = real(Bk'*(transpose(A).*M)/n^2);
        
        H4 = real(transpose(mat1.BkdBW')*mat.BwdBK'/n);
        
        ddf_wk1 = real(4*H2+2*H4)';
        
        ddf1(start_id:end_id, start_id1:end_id1) = [ddf_ww1 ddf_wk1; ddf_kw1 ddf_kk1];
        
        start_id1 = end_id1+1;
    end
    start_id = end_id+1;
    mats{mm} = [];
end

ddf = real(ddf+ddf1+ddf1');
df = real(df);

% %% smoothing
% if opt.smoothing ~= 0
%     phi = opt.phi;
%     LO = opt.LO;
%     LOOL = opt.LOOL;
%
%     start_id = 1;
%     for mm = 1:nModel
%         dw = dws(mm);
%         end_id = start_id+dk+dw-1;
%         var = V(start_id:end_id);
%         w = var(1:dw);
%         l = LO * w;
%
%         if opt.smoothing==2
%             mm
%             phi = phi/norm(w);
%         end
%
%         f = f + sum(l.^2)*phi/2;
%         df(start_id:start_id+dw-1) = df(start_id:start_id+dw-1) + phi * LOOL * w;
%         ddf(start_id:start_id+dw-1, start_id:start_id+dw-1) = ddf(start_id:start_id+dw-1, start_id:start_id+dw-1) + phi * LOOL;
%         start_id = end_id+1;
%     end
%
% end

%% smoothing
if ~(~isfield(opt, 'smoothing') || opt.smoothing == 0)
    invKse = opt.invKse*opt.lambda_w;
    if opt.smoothk
        invKse1 = opt.invKse1*opt.lambda_k;
    end
    
    start_id = 1;
    for mm = 1:nModel
        dw = dws(mm);
        dk = dks(mm);
        
        end_id = start_id+dk+dw-1;
        var = V(start_id:end_id);
        w = var(1:dw);
        if opt.smoothing == 2
            w = w/norm(w);
        end
        f = f + w'*invKse*w/2;
        df(start_id:start_id+dw-1) = df(start_id:start_id+dw-1) + invKse*w;
        ddf(start_id:start_id+dw-1, start_id:start_id+dw-1) = ddf(start_id:start_id+dw-1, start_id:start_id+dw-1) + invKse;
        
        if opt.smoothk
            k = var(1+dw:end);
            f = f + k'*invKse1*k/2;
            df(start_id+dw:end_id) = df(start_id+dw:end_id) + invKse1*k;
            ddf(start_id+dw:end_id, start_id+dw:end_id) = ddf(start_id+dw:end_id, start_id+dw:end_id) + invKse1;
        end
        
        start_id = end_id+1;
    end
end
