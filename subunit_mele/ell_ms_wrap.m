function [var_ell, w_ell, k_ell, C_ell, b_ell, a_ell] = ell_ms_wrap(var_init, Phi, invPhi, sta, STC, opt,C_bstc,b_bstc,a_bstc, x, y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MELE, moment-based estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

floss = @(var) ell_ms_fast(var, opt, Phi, invPhi, sta, STC);

if floss(var_init)>1e50 & strcmp(opt.sub, 'sigm')
    display('Use the input var, but rescale var when the value is too large for sigmoid nonlinearity.')
    var_init = var_init*1e-2;
end
if floss(var_init)>1e50
    display('Use Wu et.al, 2015 to initialize var instead of using the input var when the value is too large.')
    var_init = subunit_init_moment(C_bstc, b_bstc, a_bstc, x, y, opt, 1);
end

if floss(var_init)>1e50
    display('Random initialization when var is still too large.')
    var_init = [];
    for mm=1:opt.nModel
        var_init = [var_init; -rand(opt.nDim_ws(mm),1); randn(opt.nDim_ks(mm),1)];
    end
end
% test Deriv (using finite differencing)
% fprintf('Now checking Hessian\n');
% HessCheck(floss, var_init);

display('Doing optimization')
options = optimset('display', 'iter', 'GradObj','on','Hessian','on','maxIter', 1e3,'largescale','on');
[var_ell,fval] = fminunc(floss, var_init, options);

[var_ell, w_ell, k_ell] = split_wk(var_ell, opt);
[f,df,ddf,C_ell,b_ell,a_ell] = floss(var_ell);
