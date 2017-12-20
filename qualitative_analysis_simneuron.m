%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Qualitative analysis for a simulated complex cell
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear,clf,addpath(genpath(pwd));

neuron = load('simulated_complex_cell.mat');
nDim_k0s = [5;5];
dTimeEmbedding = neuron.dTimeEmbedding; % number of time bins to include in temporal kernel
xDim = neuron.xDim; % dimension of stimuli
nx = xDim * dTimeEmbedding; % total dimension of stimuli
x = neuron.x; % stimulus design matrix
y = neuron.y; % spike train

%% Subunit, two neurons
%% Set options
opt.nSamples = size(x,1); % number of training samples
opt.nSpikes = sum(y); % number of spikes in the training set
opt.nModel = length(nDim_k0s); % number of models, 1 model for single cell and multiple models for complex cell
opt.nkt = dTimeEmbedding;
opt.plotfig = 1; % plot figure flag
opt.nDim0 = xDim; % dimension of the raw stimulus
opt.nDim = opt.nkt*opt.nDim0; % dimension of the design matrix
opt.nonl = 'exp'; % output nonlinearity
opt.sub = 'quad'; % subunit nonlinearity
opt.initid = 5; % flag for initialization of parameters
opt.shift = 1; % bias in sigmoid nonlinearity
opt.init_ls = 0; % whether initialize MELE with MLS solution or not

opt.nDim_k0s = nDim_k0s;
opt.nDim_ks = zeros(opt.nModel,1);
opt.nDim_ws = zeros(opt.nModel,1);
for mm = 1:opt.nModel
    nDim_k0 = nDim_k0s(mm); % dimension of k for the raw stimulus
    opt.nDim_ks(mm) = nDim_k0*opt.nkt; % dimension of k for the design matrix
    opt.nDim_ws(mm) = opt.nDim0-nDim_k0+1; % dimension of w
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   1. BSTC (Park et.al, 2011)  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Estimate STC
fprintf('Computing STC...\n');
[sta, STC, rawmu, rawcov] = simpleSTC(x, y, 1); % train

% BSTC
Phi = rawcov;
[Phi1, invPhi] = svd_inv(Phi, 1e-4);
[STC1, invSTC] = svd_inv(STC, 1e-4);
b_bstc = invSTC*sta';
C_bstc = invPhi - invSTC;
[uu, ss, vv]=svd(C_bstc);
ds = diag(ss);
cds = cumsum(ds);
id = cds<cds(end)*0.8;
C_bstc = uu(:,id)*ss(id,id)*vv(:,id)';
a_bstc = log(opt.nSpikes/opt.nSamples*det(invSTC*Phi)^(0.5)*exp(-sta*invPhi*invSTC*sta'/2));

%% smoothing
opt.smoothing = 0; % flag for smoothing
opt.rho = 1; % marginal variance of smoothing kernel for w
opt.d = 3; % length scale of smoothing kernel for w
opt.lambda_w = 1e-1; % penalty parameter
opt.smoothk = 0; % flag for smoothing k, usually k doesn't need a smooth prior
opt.rho1 = 1; % marginal variance of smoothing kernel for k
opt.d1 = 3; % length scale of smoothing kernel for k
opt.lambda_k = 1e-3; % penalty parameter
opt.cv_sm = 0; % whether or not doing cross validation or hold-out test to find better smoothing hyperparameters
if ~(~isfield(opt, 'smoothing') || opt.smoothing == 0)
    opt = setopt_smooth(opt); % store the fourier basis and spectral kernel given a set of hyperparameters
    if opt.cv_sm
        % use cross validation or hold-out set to find better hyperparameters for smoothing
        % kernels for w and k. For each set of hyperparameters, use BSTC to find
        % a solution for w and k first, then use the square loss objective
        % function combined with smoothing priors for w and k to find smooth
        % w and k. The log likelihood on the test set will be evaluated to find
        % better hyperparameters. What we care about in this part is the set of
        % better hyperparameters for smoothing w and k.
        
        opt = cv_smooth(x, y, opt);
        % opt = holdout_smooth(x, y, opt);
    end
end

%% Find a way to initialize
switch opt.initid
    case 1 % use true value plus a very small noise; can't use it for real data
        var_init = params.var0+randn(size(params.var0))*1e-2;
    case 2 % random initialization for w and k together
        var_init = randn(sum(opt.nDim_ws)+sum(opt.nDim_ks),1);
    case 3 % random initialization for w and k separately
        var_init = [];
        for mm=1:opt.nModel
            var_init = [var_init; -rand(opt.nDim_ws(mm),1); randn(opt.nDim_ks(mm),1)];
        end
    case 4 % Vintch's initialization in Vintch et.al, 2012
        [var_init, kvintch1] = vintch_init(x, y, opt.nDim0, opt.nDim_k0s(1), STC, opt.nModel);
    case 5 % the initialization described in section 5.1 in Wu et.al, 2015
        var_init = subunit_init_moment(C_bstc, b_bstc, a_bstc, x, y, opt, 1);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            3. MELE            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum expected log-likelihood estimator
tic
[var_ell, w_ell, k_ell, C_ell, b_ell, a_ell] = ell_ms_wrap(var_init, Phi, invPhi, sta, STC, opt, C_bstc, b_bstc, a_bstc, x, y);
t_ell = toc;

%%
w_ell1 = w_ell;
k_ell1 = k_ell;

figure(1),clf
subplot(221), plot(w_ell1(:,1)); title('ell w1')
subplot(222), plot(w_ell1(:,2)); title('ell w2')
subplot(223), imagesc(reshape(k_ell1(:,1),opt.nkt,[])); title('ell k1')
subplot(224), imagesc(reshape(k_ell1(:,2),opt.nkt,[])); title('ell k2')
colormap bone

%% Generate y given estimations
if 1
    thetas = cell(opt.nModel,1);
    for mm = 1:opt.nModel
        optmm.nDim = opt.nDim;
        optmm.nDim_k = opt.nDim_ks(mm);
        optmm.nDim_w = opt.nDim_ws(mm);
        optmm.nkt = opt.nkt;
        w = w_ell1(:,mm);
        k = k_ell1(:,mm);
        [~, K] = genCK(w, k, optmm); % get Toeplitz matrix K from k
        bases = K';
        
        % collect into structure theta
        thetas{mm}.b = bases*w;
        thetas{mm}.C = bases * diag(w) * bases'; % C = K'*diag(w)*K
        thetas{mm}.Ds = sign(w);
        thetas{mm}.W = bases * diag(sqrt(abs(w))); % W = K'*sqrt(diag(abs(w)))
        thetas{mm}.a = a_ell;
        thetas{mm}.K = K;
        thetas{mm}.k = k;
        thetas{mm}.w = w;
    end
    fnl = @(x) exp(x);
    f = @(t,x) (t.a + x * t.b + .5 * ((x * t.W).^2) * t.Ds);
    
    fx = zeros(size(x,1),1);
    for mm = 1:opt.nModel
        fx = fx+f(thetas{mm}, x);
    end
else
    fnl = @(x) exp(x);
    f = @(t,x) (t.a+x*t.b+.5*sum((x*t.C).*x,2));
    thetas = [];
    thetas.a = a_ell;
    thetas.b = b_ell;
    thetas.C = C_ell;
    fx = f(thetas, x);
end
fx = fnl(fx);
y_sim = poissrnd(fx);
nSpikes_sim = sum(y_sim);
st_sim = find(y_sim > 0); st_sim = st_sim(:); % column vector of spike indices
sps_sim = nSpikes_sim / opt.nSamples;
maxY_sim = max(y_sim);
fprintf('Total [%d] spikes, max spikes per bin [%d], average firing rate [%f] per bin\n', nSpikes_sim, maxY_sim, sps_sim);
figure(2)
subplot(211),plot(y),title('true spikes')
subplot(212),plot(y_sim),title('simulated spikes')

%% % Estimate STA and STC
figure(3)
[sta_neuron, STC_neuron, rawmu, rawcov] = simpleSTC(x, y, 1); % train
[sta_sim, STC_sim, rawmu_sim, rawcov_sim] = simpleSTC(x, y_sim, 1); % train

sta_neuron = reshape(sta_neuron,dTimeEmbedding,[]);
sta_sim = reshape(sta_sim,dTimeEmbedding,[]);
subplot(461),imagesc(sta_neuron),title('neuron: STA'), axis off, box off
subplot(467),imagesc(sta_sim),title('simulated data: STA'), axis off, box off

[u1,s1,~] = svd(STC_neuron);
[u2,s2,~] = svd(STC_sim);

c = 1;
for ii=size(STC_neuron,1)-3:size(STC_neuron,1)
    stc1 = reshape(u1(:,ii),dTimeEmbedding,[]);
    subplot(4,6,c+1),imagesc(stc1),axis off, box off
    title(['suppressive STC filter', num2str(c)])
    c = c+1;
end

c = 1;
for ii=size(STC_sim,1)-3:size(STC_sim,1)
    stc1 = reshape(u2(:,ii),dTimeEmbedding,[]);
    subplot(4,6,c+7),imagesc(stc1),axis off, box off
    title(['suppressive STC filter', num2str(c)])
    c = c+1;
end

c = 1;
for ii=1:6
    stc1 = reshape(u1(:,ii),dTimeEmbedding,[]);
    subplot(4,6,c+12),imagesc(stc1),axis off, box off
    title(['excitatory STC filter', num2str(c)])
    c = c+1;
end

c = 1;
for ii=1:6
    stc1 = reshape(u2(:,ii),dTimeEmbedding,[]);
    subplot(4,6,c+18),imagesc(stc1),axis off, box off
    title(['excitatory STC filter', num2str(c)])
    c = c+1;
end

colormap bone

subplot(4,6,6),plot(diag(s1))
subplot(4,6,12),plot(diag(s2))
