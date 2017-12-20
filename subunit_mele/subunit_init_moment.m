function var_init = subunit_init_moment(CML, bML, aML, x, y, opt, iter)
nDim_w = opt.nDim_ws(1);
nDim_k = opt.nDim_ks(1);
nDim = opt.nDim;
nModel = opt.nModel;
w = normpdf(1:nDim_w,round(nDim_w/2),15)';
w = repmat(w, 1, nModel);
k = randn(nDim_k, nModel); k = orth(k);
w_new = w;
k_new = k;
WKK = fft(fft(CML)');
% WKK(abs(WKK)<1e-10) = 0;
% iter = 10;
difw = [];
difk = [];
for i=1:iter
    w_old = w_new;
    k_old = k_new;
    
    %% fix w, find k
    A = sqrt(sum(w_new.^2,1));
    A = sign(w_new(1,:)./A).*A;
    w0 = w_new(:,1)/A(1);
    w_pz_nkt = [w0(1:end-1) zeros(size(w0,1)-1, opt.nkt-1)]';
    w_pz_nkt = [zeros(nDim_k-1,1); w_pz_nkt(:); w0(end)];
    wh = fft(w_pz_nkt,nDim);
    Mw = circulant1(wh,1);
    Mw(abs(Mw)<1e-10) = 1;
    KK = transpose(WKK./Mw);
    KK(isnan(KK)) = 0;
    kk = real(ifft(ifft(KK)'));
    [uu, ss, vv] = svd(kk(1:nDim_k, 1:nDim_k));
    k_new = uu(:, 1:nModel);
    alpha_new = diag(ss(1:nModel, 1:nModel));
    alpha_new = (sign(k_new(1,:)).*sign(vv(1,1:nModel)))'.*alpha_new;
    
    %% fix k, find w
    kh = fft(k_new,nDim);
    KK = transpose(kh*diag(alpha_new)*kh');
    KK(abs(KK)<1e-10) = 1;
    Mw = WKK./KK;
    Mw(isnan(Mw)) = 0;
    whh = cleanup_circ(Mw);
    w1 = real(ifft(whh));
    w_new = w1(nDim_k:end);
    w_new1 = reshape([w_new; zeros(opt.nkt-1, 1)], opt.nkt, []);
    w_new = w_new1(1,:)';
    w_new = bsxfun(@times, repmat(w_new, 1, nModel), alpha_new');
    
    %% smoothing
    if ~(~isfield(opt, 'smoothing') || opt.smoothing == 0)
        w_new_fft = fft(w_new, length(opt.kdiag));
        w_new_fft = w_new_fft.*repmat(sqrt(opt.kdiag),1,nModel);
        w_new = abs(ifft(w_new_fft, size(w_new,1)));
        if opt.smoothk
            k_new_fft = fft(k_new, length(opt.kdiag1));
            k_new_fft = k_new_fft.*repmat(sqrt(opt.kdiag1),1,nModel);
            k_new = abs(ifft(k_new_fft, size(k_new,1)));
            k_new = orth(k_new);
        end
    end
    %% collect dif
    difw = [difw; norm(w_old-w_new, 'fro')];
    difk = [difk; norm(k_old-k_new, 'fro')];
end
var_init = [w_new; k_new];
var_init = var_init(:);

% optmm = opt;
% optmm.nDim_k = opt.nDim_ks(1);
% optmm.nDim_w = opt.nDim_ws(1);
% [C1, K1] = genCK(w, k, optmm);
% % %% gen C from k and w in Fourier domain
% A = sqrt(sum(w.^2,1));
% A = sign(w(1,:)./A).*A;
% w0 = w(:,1)/A(1);
% kh = fft(k,optmm.nDim);
% KK = transpose(kh*diag(A)*kh');
% w_pz_nkt = [w0(1:end-1) zeros(size(w0,1)-1, opt.nkt-1)]';
% w_pz_nkt = [zeros(optmm.nDim_k-1,1); w_pz_nkt(:); w0(end)];
% wh = fft(w_pz_nkt,optmm.nDim);
% Mw = circulant1(wh,1);
% WKK = Mw.*KK;
% C2 = real(ifft(ifft(WKK)'));
