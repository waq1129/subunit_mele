function [f, df, ddf, C, b, a] = ell_ms_fast(V, opt, Phi, invPhi, sta, STC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MELE, moment-based estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = sta';
Lambda = STC;

%% unpack variables
nkt = opt.nkt;
n = opt.nDim;
nDim_ws = opt.nDim_ws;
nDim_ks = opt.nDim_ks;
nModel = opt.nModel;
nDim = opt.nDim;
B = fft(eye(nDim));

C = zeros(n,n);
b =  zeros(n,1);
start_id = 1;
for mm = 1:nModel
    nDim_k = nDim_ks(mm);
    nDim_w = nDim_ws(mm);
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

Phi_C = invPhi-C; % Phi^{-1}-K'*diag(w)*K
I_PhiC = eye(nDim)-Phi*C; % I-Phi*K'*diag(w)*K
I_PhiCinv = I_PhiC\eye(nDim);
% I_PhiCPhi = I_PhiCinv*Phi;
[~, I_PhiCPhi] = svd_inv(Phi_C, 1e-4);

%% objective
[R, p] = chol(Phi_C);
dett = det(Phi_C);
if p~=0 || dett<=0 % if not pd, then return 1e100
    f = 1e100;
    df = zeros(length(V),1);
    ddf = zeros(length(V), length(V));
    a = 0;
    return;
end
logdetterm = logdetns(Phi_C);
f = trace(C*(Lambda+mu*mu'))/2+b'*mu+logdetterm/2-b'*I_PhiCPhi*b/2;
f = -f;

%% hessian
dC = Lambda-I_PhiCPhi;
df = zeros(size(V));
ddf = zeros(size(V,1),size(V,1));
ddf1 = zeros(size(V,1),size(V,1));

start_id = 1;
nDim2 = nDim^2;
nDim4 = nDim^4;

I_PhiCPhiB = I_PhiCPhi*B';
BPhiB = B*I_PhiCPhiB;
BLambdamuPhiB = B*(Lambda+mu*mu')*B'-BPhiB;
BPhib = I_PhiCPhiB'*b;
Bmu = B*mu;
BPhibt = transpose(BPhib)';

mats = cell(nModel,1);
for mm = 1:nModel
    nDim_k = nDim_ks(mm);
    nDim_w = nDim_ws(mm);
    end_id = start_id+nDim_k+nDim_w-1;
    var = V(start_id:end_id);
    mat.w = var(1:nDim_w);
    mat.k = var(1+nDim_w:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    
    %% dictionary
    mat.K = genToeplitz(mat.k, opt_mm);
    mat.Kmu = mat.K*mu;
    wid = nDim_k:nkt:nDim;
    mat.Bw = B(:,wid);
    mat.Bk = B(:,1:nDim_k);
    mat.BWB = bsxfun(@times, mat.Bw, mat.w')*mat.Bw';
    mat.BK = mat.Bk*mat.k;
    mat.BW = mat.Bw*mat.w;
    mat.Bkt = mat.Bk';
    mat.Bktt = transpose(mat.Bkt);
    mat.BkBPhib = bsxfun(@times, mat.Bkt, BPhib');
    mat.KPhi = mat.K*I_PhiCPhi;
    mat.KPhib = mat.KPhi*b;
    mat.BWBdBK = bsxfun(@times, mat.BWB, transpose(mat.BK));
    mat.BWBdBKBPhiB = mat.BWBdBK*BPhiB;
    mat.BPhiBdBW = bsxfun(@times, BPhiB, transpose(mat.BW));
    mat.BwdBK = bsxfun(@times, mat.Bw', transpose(mat.BK));
    mat.BdBPhibB = mat.BkBPhib*mat.Bw;
    mat.BwdBKBPhib = mat.BwdBK*BPhib;
    mat.BWBdBKBPhib = mat.BWBdBK*BPhib;
    mat.BkdBWBdBKBPhib = bsxfun(@times, mat.Bkt, transpose(mat.BWBdBKBPhib))/nDim2;
    mat.dBPhibBWBdBKBPhiB = bsxfun(@times, BPhibt, mat.BWBdBKBPhiB);
    
    BWBdBKBPhiBdBK = bsxfun(@times, mat.BWBdBKBPhiB, mat.BK');
    BWBdBKBPhiBBwdBK = mat.BWBdBKBPhiB*mat.BwdBK'/nDim;
    BkBPhibBWBdBKBPhiBBwdBK = mat.BkBPhib*BWBdBKBPhiBBwdBK;
    BwdBKBPhiBdBW = mat.BwdBK*mat.BPhiBdBW;
    
    %% gradient
    
    %% nDim_w
    g1 = diag(mat.K*dC*mat.K')/2;
    g2 = (mat.Kmu+1).^2/2;
    g3 = (mat.KPhib+1).^2/2;
    df_w = g1+g2-g3;
    
    %% nDim_k
    % traceterm = trace(BLambdamuPhiB*diag(mat.BK)'*mat.BWBdBK)/nDim2/2
    % K0 = mat.BwdBK*B/nDim;
    
    A = bsxfun(@times, BLambdamuPhiB, mat.BK')*mat.BWB;
    g1 = mat.Bkt*transpose(diag(A))'/nDim2;
    g2 = mat.Bkt*(transpose(Bmu').*mat.BW)/nDim;
    g3 = mat.Bkt*transpose(b'*bsxfun(@times, I_PhiCPhiB, transpose(mat.BW)))/nDim;
    g4 = mat.Bkt*(mat.BWBdBKBPhib.*BPhibt)/nDim2;
    df_k = real(g1+g2-g3-g4);
    
    df(start_id:end_id) = [df_w; df_k];
    
    %% hessian
    
    %% ww
    KPhiK = mat.KPhi*mat.K';
    
    H1 = -KPhiK.^2/2;
    H2 = bsxfun(@times, mat.KPhib+1, (KPhiK+bsxfun(@times, KPhiK, mat.KPhib')));
    ddf_ww = H1-H2;
    
    %% kk
    H1 = real(mat.Bkt*(transpose(BLambdamuPhiB).*mat.BWB)*mat.Bk/nDim2);
    
    A = BPhiB;
    M = BWBdBKBPhiBdBK*mat.BWB;
    H2 = real(mat.Bkt*(transpose(A).*M)*mat.Bk/nDim4);
    
    A = mat.BWBdBKBPhiB;
    M = mat.BWBdBKBPhiB;
    H3 = real(mat.Bkt*(transpose(A).*M)*mat.Bktt/nDim4);
    
    Ha = H1-H2-H3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    A = mat.BPhiBdBW;
    M = mat.BW';
    H4 = real(mat.Bkt*(bsxfun(@times, transpose(A), M))*mat.Bk/nDim2);
    
    A = bsxfun(@times, mat.BWBdBKBPhiB, transpose(mat.BW));
    M = BPhib';
    H5 = real(mat.Bkt*(bsxfun(@times, transpose(A), M))*mat.Bktt/nDim^3);
    
    A = mat.BPhiBdBW;
    M = mat.BWBdBKBPhib';
    H6 = real(mat.Bkt*(bsxfun(@times, transpose(A), M))*mat.Bk/nDim^3);
    
    Hb = H4+H5+H6;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    A = bsxfun(@times, mat.BWBdBKBPhib, transpose(bsxfun(@times, BPhiB, transpose(mat.BWBdBKBPhib)))');
    H7 = real(mat.Bkt*A*mat.Bk/nDim4);
    
    A = bsxfun(@times, mat.BWBdBKBPhib, transpose(bsxfun(@times, mat.BWBdBKBPhiB', transpose(BPhib)))');
    H8 = real(mat.Bkt*A*mat.Bktt/nDim4);
    
    A = bsxfun(@times, mat.BWBdBKBPhib, transpose(mat.BPhiBdBW)');
    H9 = real(mat.Bkt*A*mat.Bk/nDim^3);
    
    A = bsxfun(@times, mat.dBPhibBWBdBKBPhiB, transpose(mat.BW));
    H10 = real(mat.Bkt*A*mat.Bktt/nDim^3);
    
    A = bsxfun(@times, mat.dBPhibBWBdBKBPhiB, mat.BK')*mat.BWB;
    M = BPhib;
    H11 = real(mat.Bkt*(bsxfun(@times, A, transpose(M)))*mat.Bk/nDim4);
    
    A = mat.dBPhibBWBdBKBPhiB;
    M = mat.BWBdBKBPhib;
    H12 = real(mat.Bkt*(bsxfun(@times, A, transpose(M)))*mat.Bktt/nDim4);
    
    A = bsxfun(@times, BPhibt, mat.BWB);
    M = BPhib;
    H13 = real(mat.Bkt*(bsxfun(@times, A, transpose(M)))*mat.Bk/nDim2);
    
    Hc = H7+H8+H9+H10+H11+H12+H13;
    
    ddf_kk = Ha-Hb-Hc;
    
    %% kw
    A = mat.BwdBK*BLambdamuPhiB;
    M = mat.Bw;
    H1 = real(mat.Bkt*(transpose(A).*M)/nDim2);
    
    A = mat.BwdBK*BPhiB;
    M = BWBdBKBPhiBdBK*mat.Bw;
    H2 = real(mat.Bkt*(transpose(A).*M)/nDim4);
    
    H3 = real(bsxfun(@times, mat.Bkt, Bmu')*mat.Bw/nDim);
    
    Ha = H1-H2+H3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    A = BwdBKBPhiBdBW;
    M = mat.BwdBKBPhib';
    H4 = real(mat.Bkt*(bsxfun(@times, transpose(A), M))/nDim^3);
    
    H5 = real(mat.BdBPhibB/nDim);
    
    A = BwdBKBPhiBdBW;
    H6 = real(mat.Bkt*transpose(A)/nDim2);
    
    Hb = H4+H5+H6;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    A = mat.BdBPhibB;
    M = mat.BwdBKBPhib/nDim2;
    H7 = bsxfun(@times, A, transpose(M));
    
    A = BkBPhibBWBdBKBPhiBBwdBK;
    M = mat.KPhib/nDim2;
    H8 = bsxfun(@times, A, transpose(M));
    
    A = mat.BkdBWBdBKBPhib*transpose(B')*mat.KPhi';
    M = b'*transpose(mat.KPhi);
    H9 = bsxfun(@times, A, M);
    
    H10 = (mat.K*I_PhiCPhiB*transpose(mat.BkdBWBdBKBPhib))';
    
    H11 = BkBPhibBWBdBKBPhiBBwdBK/nDim2;
    
    Hc = H7+H8+H9+H10+H11;
    
    ddf_kw = real(Ha-Hb-Hc);
    
    mats{mm} = mat;
    %%
    
    ddf(start_id:end_id, start_id:end_id) = [ddf_ww ddf_kw'; ddf_kw ddf_kk];
    
    start_id = end_id+1;
    
end

start_id = 1;
for mm = 1:nModel
    mat = mats{mm};
    nDim_k = nDim_ks(mm);
    nDim_w = nDim_ws(mm);
    end_id = start_id+nDim_k+nDim_w-1;
    Bw = mat.Bw;
    Bk = mat.Bk;
    Bkt = mat.Bkt;
    Bktt = mat.Bktt;
    
    start_id1 = end_id+1;
    
    for nn = mm+1:nModel
        nDim_k1 = nDim_ks(nn);
        nDim_w1 = nDim_ws(nn);
        end_id1 = start_id1+nDim_k1+nDim_w1-1;
        mat1 = mats{nn};
        
        %% dictionary
        
        BwdBK1BPhiBdBW = mat1.BwdBK*mat.BPhiBdBW;
        BwdBKBPhiBdBW1 = mat.BwdBK*mat1.BPhiBdBW;
        
        BWBdBKBPhiBBwdBK1 = mat.BWBdBKBPhiB*mat1.BwdBK'/nDim;
        BW1BdBK1BPhiBBwdBK = mat1.BWBdBKBPhiB*mat.BwdBK'/nDim;
        
        BW1BdBK1BPhiBdBK = bsxfun(@times, mat1.BWBdBKBPhiB, mat.BK');
        BWBdBKBPhiBdBK1 = bsxfun(@times, mat.BWBdBKBPhiB, mat1.BK');
        
        BkBPhibBW1BdBK1BPhiBBwdBK = mat.BkBPhib*BW1BdBK1BPhiBBwdBK;
        BkBPhibBWBdBKBPhiBBwdBK1 = mat.BkBPhib*BWBdBKBPhiBBwdBK1;
        
        %% hessian
        
        %% ww
        KPhiK1 = mat.KPhi*mat1.K';
        
        H1 = -KPhiK1.^2/2;
        
        H2 = bsxfun(@times, mat.KPhib+1, (KPhiK1+bsxfun(@times, KPhiK1, mat1.KPhib')));
        
        ddf_ww1 = H1-H2;
        
        %% kk
        
        A = BPhiB;
        M = BWBdBKBPhiBdBK1*mat1.BWB;
        H2 = real(Bkt*(transpose(A).*M)*Bk/nDim4);
        
        A = mat1.BWBdBKBPhiB;
        M = mat.BWBdBKBPhiB;
        H3 = real(Bkt*(transpose(A).*M)*Bktt/nDim4);
        
        Ha = -H2-H3;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = mat.BPhiBdBW;
        M = mat1.BW';
        H4 = real(Bkt*(bsxfun(@times, transpose(A), M))*Bk/nDim2);
        
        A = bsxfun(@times, mat1.BWBdBKBPhiB, transpose(mat.BW));
        M = BPhib';
        H5 = real(Bkt*(bsxfun(@times, transpose(A), M))*Bktt/nDim^3);
        
        A = mat.BPhiBdBW;
        M = mat1.BWBdBKBPhib';
        H6 = real(Bkt*(bsxfun(@times, transpose(A), M))*Bk/nDim^3);
        
        Hb = H4+H5+H6;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = bsxfun(@times, mat.BWBdBKBPhib, transpose(bsxfun(@times, BPhiB, transpose(mat1.BWBdBKBPhib)))');
        H7 = real(Bkt*A*Bk/nDim4);
        
        A = bsxfun(@times, mat.BWBdBKBPhib, transpose(bsxfun(@times, mat1.BWBdBKBPhiB', transpose(BPhib)))');
        H8 = real(Bkt*A*Bktt/nDim4);
        
        A = bsxfun(@times, mat.BWBdBKBPhib, transpose(mat1.BPhiBdBW)');
        H9 = real(Bkt*A*Bk/nDim^3);
        
        A = bsxfun(@times, mat.dBPhibBWBdBKBPhiB, transpose(mat1.BW));
        H10 = real(Bkt*A*Bktt/nDim^3);
        
        A = bsxfun(@times, mat.dBPhibBWBdBKBPhiB, mat1.BK')*mat1.BWB;
        M = BPhib;
        H11 = real(Bkt*(bsxfun(@times, A, transpose(M)))*Bk/nDim4);
        
        A = mat.dBPhibBWBdBKBPhiB;
        M = mat1.BWBdBKBPhib;
        H12 = real(Bkt*(bsxfun(@times, A, transpose(M)))*Bktt/nDim4);
        
        Hc = H7+H8+H9+H10+H11+H12;
        
        ddf_kk1 = Ha-Hb-Hc;
        
        
        %% kw1
        
        A = mat1.BwdBK*BPhiB;
        M = BWBdBKBPhiBdBK1*Bw;
        H2 = real(Bkt*(transpose(A).*M)/nDim4);
        
        Ha = -H2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = BwdBK1BPhiBdBW;
        M = mat1.BwdBKBPhib';
        H4 = real(Bkt*(bsxfun(@times, transpose(A), M))/nDim^3);
        
        A = BwdBK1BPhiBdBW;
        H6 = real(Bkt*transpose(A)/nDim2);
        
        Hb = H4+H6;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = BkBPhibBWBdBKBPhiBBwdBK1;
        M = mat1.KPhib/nDim2;
        H8 = bsxfun(@times, A, transpose(M));
        
        A = mat.BkdBWBdBKBPhib*transpose(B')*mat1.KPhi';
        M = b'*transpose(mat1.KPhi);
        H9 = bsxfun(@times, A, M);
        
        H10 = (mat1.K*I_PhiCPhiB*transpose(mat.BkdBWBdBKBPhib))';
        
        H11 = BkBPhibBWBdBKBPhiBBwdBK1/nDim2;
        
        Hc = H8+H9+H10+H11;
        
        ddf_kw1 = real(Ha-Hb-Hc);
        
        
        %% wk1
        A = mat.BwdBK*BPhiB;
        M = BW1BdBK1BPhiBdBK*Bw;
        H2 = real(Bkt*(transpose(A).*M)/nDim4);
        
        Ha = -H2;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = BwdBKBPhiBdBW1;
        M = mat.BwdBKBPhib';
        H4 = real(Bkt*(bsxfun(@times, transpose(A), M))/nDim^3);
        
        A = BwdBKBPhiBdBW1;
        H6 = real(Bkt*transpose(A)/nDim2);
        
        Hb = H4+H6;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        A = BkBPhibBW1BdBK1BPhiBBwdBK;
        M = mat.KPhib/nDim2;
        H8 = bsxfun(@times, A, transpose(M));
        
        A = mat1.BkdBWBdBKBPhib*transpose(B')*mat.KPhi';
        M = b'*transpose(mat.KPhi);
        H9 = bsxfun(@times, A, M);
        
        H10 = (mat.K*I_PhiCPhiB*transpose(mat1.BkdBWBdBKBPhib))';
        
        H11 = BkBPhibBW1BdBK1BPhiBBwdBK/nDim2;
        
        Hc = H8+H9+H10+H11;
        
        ddf_wk1 = real(Ha-Hb-Hc)';
        
        ddf1(start_id:end_id, start_id1:end_id1) = [ddf_ww1 ddf_wk1; ddf_kw1 ddf_kk1];
        
        start_id1 = end_id1+1;
    end
    mats{mm} = [];
    start_id = end_id+1;
    
end

ddf = -real(ddf+ddf1+ddf1');
df = -real(df);
%% a and b
a = -log(opt.nSamples/opt.nSpikes*det(I_PhiC)^(-0.5)*exp(b'*(Phi_C\b)/2));

% %% smoothing
% if opt.smoothing ~= 0
%     phi = opt.phi;
%     LO = opt.LO;
%     LOOL = opt.LOOL;
%
%     start_id = 1;
%     for mm = 1:nModel
%         nDim_w = nDim_ws(mm);
%         end_id = start_id+nDim_k+nDim_w-1;
%         var = V(start_id:end_id);
%         w = var(1:nDim_w);
%         l = LO * w;
%
%         if opt.smoothing==2
%             phi = phi/norm(w);
%         end
%
%         f = f + sum(l.^2)*phi/2;
%         df(start_id:start_id+nDim_w-1) = df(start_id:start_id+nDim_w-1) + phi * LOOL * w;
%         ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) = ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) + phi * LOOL;
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
        nDim_w = nDim_ws(mm);
        nDim_k = nDim_ks(mm);
        
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
