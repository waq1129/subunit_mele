function C = formC_Fourier(w, k, opt)

kh = fft(k,opt.nDim);
KK = transpose(kh*kh');
w_pz_nkt = [w(1:end-1) zeros(size(w,1)-1, opt.nkt-1)]';
w_pz_nkt = [zeros(opt.nDim_k-1,1); w_pz_nkt(:); w(end)];
wh = fft(w_pz_nkt,opt.nDim);
Mw = circulant1(wh,1);
WKK = Mw.*KK;
C = real(ifft(ifft(WKK)'));