function K1 = formK_Fourier(k, opt)

kh = fft(k,opt.nDim);
KK = real(ifft(ifft(diag(kh))')')*opt.nDim;
wid = opt.nDim_k:opt.nkt:opt.nDim;
K1 = KK(wid,:);
K1(abs(K1)<1e-10) = 0;