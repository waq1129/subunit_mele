function [loglikeli, runtime, params_est, opt] = subunit_mele(x_train,y_train,x_test,y_test,nDim_k0s,xDim,dTimeEmbedding,sm_flag)
%% Set options
opt.nSamples = size(x_train,1); % number of training samples
opt.nSamples_test = size(x_test,1); % number of test samples
opt.nSpikes = sum(y_train); % number of spikes in the training set
opt.nSpikes_test = sum(y_test); % number of spikes in the test set
opt.nModel = length(nDim_k0s); % number of models, 1 model for single cell and multiple models for complex cell
opt.nkt = dTimeEmbedding; % number of time bins to include in temporal kernel
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
[sta, STC, rawmu, rawcov] = simpleSTC(x_train, y_train, 1); % train
[sta_test, STC_test, rawmu_test, rawcov_test] = simpleSTC(x_test, y_test, 1); % test

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

% Draw comparisons of b and C
if opt.plotfig
    figure(1),clf
    % compare b
    subplot(3,3,1:3); cla; hold all; plot(sta); grid on;
    plot(b_bstc); drawnow
    
    % compare C
    subplot(3,3,4); imagesc(STC); colorbar; title('STC'); drawnow
    subplot(3,3,5); imagesc(C_bstc); colorbar; title('C\_bstc'); drawnow
end

%% smoothing
opt.smoothing = sm_flag; % flag for smoothing
opt.rho = 1; % marginal variance of smoothing kernel for w
opt.d = 3; % length scale of smoothing kernel for w
opt.lambda_w = 1e-1; % penalty parameter
opt.smoothk = 1; % flag for smoothing k, usually k doesn't need a smooth prior
opt.rho1 = 1; % marginal variance of smoothing kernel for k
opt.d1 = 3; % length scale of smoothing kernel for k
opt.lambda_k = 1e-5; % penalty parameter
opt.cv_sm = 1; % whether or not doing cross validation or hold-out test to find better smoothing hyperparameters
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
        
        opt = cv_smooth(x_train, y_train, opt);
        % opt = holdout_smooth(x_train, y_train, opt);
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
        [var_init, kvintch1] = vintch_init(x_train, y_train, opt.nDim0, opt.nDim_k0s(1), STC, opt.nModel);
    case 5 % the initialization described in section 5.1 in Wu et.al, 2015
        var_init = subunit_init_moment(C_bstc, b_bstc, a_bstc, x_train, y_train, opt, 1);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            2. MSL             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimizing |C_bstc-K'*diag(w)*K|+|b_bstc-K'*w|
tic
[var_ls, w_ls, k_ls, C_ls, b_ls, a_ls] = msl_ms_wrap(var_init, C_bstc, b_bstc, Phi, invPhi, opt);
t_ls = toc;
if opt.plotfig
    % compare b and C
    subplot(3,3,1:3); hold on; plot(b_ls,'m'); drawnow
    subplot(3,3,6); imagesc(C_ls); colorbar; title('C\_ls'); drawnow
end
if opt.init_ls
    var_init = var_ls;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            3. MELE            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum expected log-likelihood estimator
tic
[var_ell, w_ell, k_ell, C_ell, b_ell, a_ell] = ell_ms_wrap(var_init, Phi, invPhi, sta, STC, opt, C_bstc, b_bstc, a_bstc, x_train, y_train);
t_ell = toc;

if opt.plotfig
    % compare b and C
    subplot(3,3,1:3); hold on; plot(b_ell,'k'); drawnow
    subplot(3,3,7); imagesc(C_ell); colorbar; title('C\_ell'); drawnow
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    4. Log Likelihood (MLE)    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% baseline model: the log-likelihood objective function is consistent with
% the true generative model of the simulated data
tic
[var_ll, w_ll, k_ll, C_ll, b_ll, a_ll] = loglikehood_ms_wrap_true(var_init, x_train, y_train, opt, sta, STC, opt.nSpikes);
t_ll = toc;

if opt.plotfig
    % compare b and C
    subplot(3,3,1:3); hold on; plot(b_ll,'c');
    legend('STA', 'b_bstc', 'b\_ls', 'b\_ell', 'b\_ll', 'Location', 'EastOutside'); drawnow
    subplot(3,3,8); imagesc(C_ll); colorbar; title('C\_ll'); drawnow
end

%% compute log likelihood in test set
if strcmp(opt.sub, 'quad')
    floss_test_BSTC = @(C, b, a) loglikehood_BSTC(C, b, a, x_test, y_test, opt.nSpikes_test, opt); %likelihood for C_bstc
    loglikeli.BSTC = floss_test_BSTC(C_bstc, b_bstc, a_bstc);
    loglikeli.ls = floss_test_BSTC(C_ls, b_ls, a_ls);
    loglikeli.ell = floss_test_BSTC(C_ell, b_ell, a_ell);
    loglikeli.ll = floss_test_BSTC(C_ll, b_ll, a_ll);
    loglikeli
end

if strcmp(opt.sub, 'sigm')
    opt1 = opt;
    BX = BxBx(x_test, opt1);
    opt1.BX = BX;
    opt1.smoothing = 0;
    floss_test_BSTC = @(var) -loglikehood_sigm(var, x_test, y_test, opt1, opt.nSpikes_test);
    loglikeli.ls = floss_test_BSTC(var_ls);
    loglikeli.ell = floss_test_BSTC(var_ell);
    loglikeli.ll = floss_test_BSTC(var_ll);
    loglikeli
end

%% collect running time
runtime.ls = t_ls;
runtime.ell = t_ell;
runtime.ll = t_ll;
runtime

%% collect estimation
params_est.w_ls = w_ls;
params_est.k_ls = k_ls;
params_est.C_ls = C_ls;
params_est.b_ls = b_ls;
params_est.a_ls = a_ls;

params_est.w_ell = w_ell;
params_est.k_ell = k_ell;
params_est.C_ell = C_ell;
params_est.b_ell = b_ell;
params_est.a_ell = a_ell;

params_est.w_ll = w_ll;
params_est.k_ll = k_ll;
params_est.C_ll = C_ll;
params_est.b_ll = b_ll;
params_est.a_ll = a_ll;

params_est.var_init = var_init;

if opt.plotfig
    figure(2),clf
    switch opt.nModel
        case 1
            subplot(311), plot([w_ls w_ell w_ll]);
            legend('ls','ell','ll'),title('w')
            subplot(312), plot([k_ls k_ell k_ll]);
            title('k')
            subplot(337), imagesc(reshape(k_ls,opt.nkt,[]));
            subplot(338), imagesc(reshape(k_ell,opt.nkt,[]));
            subplot(339), imagesc(reshape(k_ll,opt.nkt,[]));
            colormap bone
        case 2
            subplot(421), plot([w_ls(:,1) w_ell(:,1) w_ll(:,1)]);
            legend('ls','ell','ll'),title('w1')
            subplot(422), plot([w_ls(:,2) w_ell(:,2) w_ll(:,2)]);
            title('w2')
            subplot(423), plot([k_ls(:,1) k_ell(:,1) k_ll(:,1)]);
            legend('ls','ell','ll'),title('k1')
            subplot(424), plot([k_ls(:,2) k_ell(:,2) k_ll(:,2)]);
            title('k2')
            subplot(437), imagesc(reshape(k_ls(:,1),opt.nkt,[])); title('ls k1')
            subplot(438), imagesc(reshape(k_ell(:,1),opt.nkt,[])); title('ell k1')
            subplot(439), imagesc(reshape(k_ll(:,1),opt.nkt,[])); title('ll k1')
            subplot(4,3,10), imagesc(reshape(k_ls(:,2),opt.nkt,[])); title('ls k2')
            subplot(4,3,11), imagesc(reshape(k_ell(:,2),opt.nkt,[])); title('ell k2')
            subplot(4,3,12), imagesc(reshape(k_ll(:,2),opt.nkt,[]));title('ll k2')
            colormap bone
    end
end
