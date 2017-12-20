function opt = setopt_smooth(opt)
% Fourier-defined SE cov

nx = opt.nDim_ws(1);
nxcirc = opt.nDim_ws(1)*1;
[kdiag,wvec] = mkCov_SE_fourier(opt.rho,opt.d,nx,nxcirc);
opt.B = realDFTbasis(nx,nxcirc,wvec);
opt.Kse = opt.B*diag(kdiag)*opt.B';  % inefficient implementation (but clear)
opt.kdiag = kdiag;
opt.invkdiag = 1./kdiag;
opt.invkdiag(opt.invkdiag>1e3) = 1e3;
opt.invKse = opt.B*diag(opt.invkdiag)*opt.B';

if opt.smoothk % smooth k
    nx = opt.nDim_ks(1);
    nxcirc = opt.nDim_ks(1)*1;
    [kdiag,wvec] = mkCov_SE_fourier(opt.rho1,opt.d1,nx,nxcirc);
    opt.B1 = realDFTbasis(nx,nxcirc,wvec);
    opt.Kse1 = opt.B1*diag(kdiag)*opt.B1';  % inefficient implementation (but clear)
    opt.kdiag1 = kdiag;
    opt.invkdiag1 = 1./kdiag;
    id = abs(opt.invkdiag1)>1e3;
    opt.invkdiag1(id) = 1e3*sign(opt.invkdiag1(id));
    opt.invKse1 = opt.B1*diag(opt.invkdiag1)*opt.B1';
end