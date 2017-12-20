function [params, opt] = gen_params_exp_sigm(opt)
opt.nModel = length(opt.nDim_k0s); % number of models/neurons
opt.nDim = opt.nDim0*opt.nkt;
opt.nDim_ks = zeros(opt.nModel,1);
opt.nDim_ws = zeros(opt.nModel,1);

%% true w and k for all models
% alpha = sign(randn)*(rand(opt.nModel,1)+0.5);
alpha = [-1; 2];
ws = []; ks = []; var0 = [];
thetas = cell(opt.nModel,1);
a0 = 0;
b0 = zeros(opt.nDim,1);
C0 = zeros(opt.nDim, opt.nDim);

% generate a dictionary for k
% Kbase = real(fft(eye(opt.nDim_k0s(1)*opt.nkt)));
nk = opt.nDim_k0s(1)*opt.nkt;
Kbase = -normpdf(1:nk,(nk+1)/2,nk/5)+normpdf(1:nk,(nk+1)/2,nk/20);
Kbase = [Kbase; -normpdf(1:nk,(nk+1)/2,nk/50)+normpdf(1:nk,(nk+1)/2,nk/10)];
Kbase = Kbase';
for mm=1:opt.nModel
    
    % generate k and w for each model
    opt.nDim_ks(mm) = opt.nDim_k0s(mm)*opt.nkt; % dimension of k
    k = Kbase(:,mm);
    % k = normpdf(1:opt.nDim_ks(mm),(opt.nDim_ks(mm)+1)/2,opt.nDim_ks(mm)/5)-normpdf(1:opt.nDim_ks(mm),(opt.nDim_ks(mm)+1)/2,opt.nDim_ks(mm)/20);
    k = normalize_x(k(:));
    opt.nDim_ws(mm) = opt.nDim0+1-opt.nDim_k0s(mm);
    w = alpha(mm)*(normpdf(1:opt.nDim_ws(mm),(opt.nDim_ws(mm)+1)/2,opt.nDim_ws(mm)/5)'-5*normpdf(1:opt.nDim_ws(mm),(opt.nDim_ws(mm)+1)/2,opt.nDim_ws(mm)/15)');
    w = 8*w-1.2;
    ws = [ws w];
    ks = [ks k];
    a = 5;
    optmm.nDim = opt.nDim;
    optmm.nDim_k = opt.nDim_ks(mm);
    optmm.nDim_w = opt.nDim_ws(mm);
    optmm.nkt = opt.nkt;
    [~, K] = genCK(w, k, optmm); % get Toeplitz matrix K from k
    bases = K';
    var0 = [var0; w; k];
    
    % collect into structure theta
    thetas{mm}.b = bases*w;
    thetas{mm}.C = bases * diag(w) * bases'; % C = K'*diag(w)*K
    thetas{mm}.Ds = sign(w);
    thetas{mm}.W = bases * diag(sqrt(abs(w))); % W = K'*sqrt(diag(abs(w)))
    thetas{mm}.a = a;
    thetas{mm}.K = K;
    thetas{mm}.k = k;
    thetas{mm}.w = w;
    
    a0 = a0+thetas{mm}.a;
    b0 = b0+thetas{mm}.b;
    C0 = C0+thetas{mm}.C;
end
params.thetas = thetas;
params.a0 = a0;
params.b0 = b0;
params.C0 = C0;
params.ks = ks;
params.ws = ws;
params.var0 = var0;
params.alpha = alpha;
