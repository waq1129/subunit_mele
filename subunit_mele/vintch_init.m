function [varvintch, kvintch] = vintch_init(x, y, n, d, STC, nm)
dw = n-d+1;
wvintch = normpdf(1:dw,round(dw/2),15)';

% wvintch = normpdf(1:dw,dw/2,dw/6);
% wvintch = wvintch-mean(wvintch);

% %% Try Vintch method (1)
% STC_all = zeros(d,d);
% nonzero_y = find(y~=0);
% nsp = sum(y);
% for ii=1:length(nonzero_y)
%     if mod(ii,round(length(nonzero_y)/10))==0
%         ii
%     end
%     xmean = x(nonzero_y(ii),:);
%     tmp = makeStimRows(xmean',dw);
%     X = tmp(dw:end,:);
%     X = bsxfun(@times, X, sqrt(abs(wvintch)));
%     Y = y(nonzero_y(ii))*sign(wvintch');
%     xSTC_i = (bsxfun(@times,X',Y)'*X')/nsp;
%     STC_all = STC_all+xSTC_i;
% end
% [uv,sv] = svd(STC_all);
% kvintch = uv(:,1);

%% Try Vintch method (2)
Cvintch = zeros(d,d);
for ii=1:dw
    Cvintch = Cvintch + wvintch(ii)*STC(ii:d+ii-1,ii:d+ii-1);
end
[uv,sv] = svd(Cvintch);
kvintch = uv(:,nm);

ws = repmat(wvintch(:), 1, nm);
ks = kvintch;

varvintch = [ws; ks];
varvintch = varvintch(:);
