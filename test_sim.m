%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test script for simulated data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear, addpath(genpath(pwd));
figure(1), clf
rng('shuffle')

%% set up options
opt.plotfig = 1; % flag for plotting figures
opt.nDim_k0s = 8; % length of true filter k
opt.nDim0 = 40; % dimension of input stimulus
opt.nkt = 1; % number of time bins to include in temporal kernel
opt.nonl = 'rec'; % output nonlinearity
opt.sub = 'quad'; % subunit nonlinearity
opt.initid = 5; % flag for initialization of parameters
opt.shift = 1; % bias in sigmoid nonlinearity
opt.init_ls = 0; % whether initialize MELE with MLS solution or not

%% generate model parameters
if strcmp(opt.nonl,'exp') && strcmp(opt.sub,'quad')
    [params, opt] = gen_params_exp_quad(opt);
end
if strcmp(opt.nonl,'rec') && strcmp(opt.sub,'quad')
    [params, opt] = gen_params_rec_quad(opt);
end
if strcmp(opt.nonl,'exp') && strcmp(opt.sub,'sigm')
    [params, opt] = gen_params_exp_sigm(opt);
end
if strcmp(opt.nonl,'rec') && strcmp(opt.sub,'sigm')
    [params, opt] = gen_params_rec_sigm(opt);
end

opt.nSamples_train = 10^3; % number of training samples
opt.nSamples_test = 10^3; % number of test samples
[x_train, y_train, x_test, y_test, opt] = gen_data_from_param(params, opt);

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
    figure(2),clf
    % compare b
    subplot(3,3,1:3); cla; hold all; plot(sta); grid on;
    plot(b_bstc); drawnow
    
    % compare C
    subplot(3,3,4); imagesc(params.C0); colorbar; title('C\_true'); drawnow
    subplot(3,3,5); imagesc(C_bstc); colorbar; title('C\_bstc'); drawnow
end

%% smoothing
opt.smoothing = 1; % flag for smoothing
opt.rho = 1; % marginal variance of smoothing kernel for w
opt.d = 3; % length scale of smoothing kernel for w
opt.lambda_w = 1e-1; % penalty parameter
opt.smoothk = 0; % flag for smoothing k, usually k doesn't need a smooth prior
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
    case 1 % use true value plus a very small noise
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

%% compare w and k with the true values.
% mse_wk.m function doesn't just compare the pairwise distance.
% It considers shifts.

clear mse var_ls1 var_ell1 var_ll1
for mm = 1:opt.nModel
    params_mm = params;
    params_mm.ws = params.ws(:,mm);
    params_mm.ks = params.ks(:,mm);
    
    [~, mse.w_ls(mm), mse.k_ls(mm), var_ls1(:,mm), w_ls1(:,mm), k_ls1(:,mm)] = mse_wk(w_ls(:,mm), k_ls(:,mm), params_mm);
    [~, mse.w_ell(mm), mse.k_ell(mm), var_ell1(:,mm), w_ell1(:,mm), k_ell1(:,mm)] = mse_wk(w_ell(:,mm), k_ell(:,mm), params_mm);
    [~, mse.w_ll(mm), mse.k_ll(mm), var_ll1(:,mm), w_ll1(:,mm), k_ll1(:,mm)] = mse_wk(w_ll(:,mm), k_ll(:,mm), params_mm);
    mse.ws = [mse.w_ls; mse.w_ell; mse.w_ll];
    mse.ks = [mse.k_ls; mse.k_ell; mse.k_ll];
end
mse.w_ls_sum = sum(mse.w_ls);
mse.w_ell_sum = sum(mse.w_ell);
mse.w_ll_sum = sum(mse.w_ll);
mse.k_ls_sum = sum(mse.k_ls);
mse.k_ell_sum = sum(mse.k_ell);
mse.k_ll_sum = sum(mse.k_ll);
mse

if opt.plotfig
    % plot w and k with normalization for each
    figure(3),clf,
    subplot(211), cla, plot([normalize_x(params.ws) normalize_x(w_ls1) normalize_x(w_ell1) normalize_x(w_ll1)]); drawnow
    legend('true','ls','ell','ll')
    title('w')
    subplot(212), cla, plot([normalize_x(params.ks) normalize_x(k_ls1) normalize_x(k_ell1) normalize_x(k_ll1)]); drawnow
    title('k')
end


