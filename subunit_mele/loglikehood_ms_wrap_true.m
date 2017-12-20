function [var_ll, w_ll, k_ll, C_ll, b_ll, a_ll] = loglikehood_ms_wrap_true(var_init, x, y, opt, sta, STC, nSpikes)
% baseline model: the log-likelihood objective function is consistent with
% the true generative model of the simulated data

if strcmp(opt.sub, 'quad') && strcmp(opt.nonl, 'exp') % log-likelihood with quadratic subunit nonlinearity and exp output nonlinearity
    floss = @(var) loglikehood_quad_exp(var, x, opt, sta, STC, nSpikes);
end

if strcmp(opt.sub, 'quad') && strcmp(opt.nonl, 'rec') % log-likelihood with quadratic subunit nonlinearity
    var_init = [var_init; rand];
    floss = @(var) loglikehood_quad(var, x, y, opt, nSpikes);
end

if strcmp(opt.sub, 'sigm') % log-likelihood with sigmoid subunit nonlinearity
    BX = BxBx(x, opt);
    opt.BX = BX;
    floss = @(var) loglikehood_sigm(var, x, y, opt, nSpikes);
end

if floss(var_init)>1e50 & strcmp(opt.sub, 'sigm')
    display('Use the input var, but rescale var when the value is too large for sigmoid nonlinearity.')
    var_init = var_init*1e-2;
end

if isnan(floss(var_init))
    display('If var_init contains Nan, re-initialize var_init with random values.')
    var_init = rand(size(var_init));
end

% test Deriv (using finite differencing)
% fprintf('Now checking Hessian\n');
% HessCheck(floss, var_init);

display('Doing optimization')
options = optimset('display', 'iter', 'GradObj','on','Hessian','on','maxIter', 1e3,'largescale','on');
[var_ll,fval] = fminunc(floss, var_init, options);

if strcmp(opt.sub, 'quad') && strcmp(opt.nonl, 'exp')
    [var_ll, w_ll, k_ll] = split_wk(var_ll, opt);
    [f,df,ddf,C_ll,b_ll,a_ll] = floss(var_ll);
end

if strcmp(opt.sub, 'quad') && strcmp(opt.nonl, 'rec')
    [var_ll(1:end-1), w_ll, k_ll] = split_wk(var_ll(1:end-1), opt);
    [f,df,ddf,C_ll,b_ll] = floss(var_ll);
    a_ll = var_ll(end);
    var_ll = var_ll(1:end-1);
end

if strcmp(opt.sub, 'sigm')
    [var_ll, w_ll, k_ll] = split_wk_nonorm(var_ll, opt);
    [C_ll, b_ll] = from_var_to_C(var_ll, opt);
    a_ll = 0;
end