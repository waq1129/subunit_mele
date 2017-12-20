function [var_new, w_new, k_new, C_new, b_new, a_new] = msl_ms_wrap(var0, C0, b0, Phi, invPhi, opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find w and k minimizing |C_bstc-K'*diag(w)*K| constrained by b_bstc=K'*w.
% The version implemented is with the form
% minimizing |C_bstc-K'*diag(w)*K|+|b_bstc-K'*w|
% It can also be implemented by replacing fminunc with fmincon and add the
% linear constraint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% msl without constraint
var = var0;
floss = @(var) msl_ms_fast(var, C0, b0, opt);

%% test Deriv (using finite differencing)
% fprintf('Now checking Hessian\n');
% HessCheck(floss,var);

display('Doing optimization')
options = optimset('display', 'off', 'GradObj','on', 'Hessian','on','maxIter', 1e3,'largescale','on');
[var_new, fval] = fminunc(floss, var, options);

%% unpack estimated w_new and k_new
[var_new, w_new, k_new] = split_wk(var_new, opt);
[f, df, ddf, C_new, b_new] = floss(var_new);

%% optimal a
Phi_C = invPhi-C_new; % Phi^{-1}-K'*diag(w)*K
I_PhiC = eye(opt.nDim)-Phi*C_new; % I-Phi*K'*diag(w)*K
a_new = -log(opt.nSamples/opt.nSpikes*det(I_PhiC)^(-0.5)*exp(b_new'*(Phi_C\b_new)/2));


