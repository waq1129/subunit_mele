function BX = BxBx(x, opt)
nDim = opt.nDim;
Bx = fft(x')/nDim;
BX = zeros(nDim^2, size(x,1));
for ii=1:nDim
    BX((ii-1)*nDim+1:ii*nDim, :) = bsxfun(@times, Bx, transpose(Bx(ii,:)'));
end
