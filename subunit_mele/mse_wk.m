function [wk, ws, ks, var0, w0, k0] = mse_wk(w, k, params)

step = max([size(w,1), size(k,1)]);
wk = inf;
ws = 0;
ks = 0;
w0 = 0;
k0 = 0;
for ii=0:step-1
    ws1 = min([mse_m(params.ws, circshift(w,ii)) mse_m(params.ws, -circshift(w,ii))]);
    [ks1, minid] = min([mse_m(params.ks, circshift(k,-ii)) mse_m(params.ks, -circshift(k,-ii))]);
    wk1 = ws1+ks1;
    if wk1<wk
        % display(['w shifts ' num2str(ii) ', k0 shifts ' num2str(-ii)])
        wk = wk1;
        ws=ws1;
        ks=ks1;
        w0 = circshift(w,ii);
        if minid==1
            k0 = circshift(k,-ii);
        else
            k0 = -circshift(k,-ii);
        end
    end
end

for ii=0:step-1
    ws1 = min([mse_m(params.ws, circshift(w,-ii)) mse_m(params.ws, -circshift(w,-ii))]);
    [ks1, minid] = min([mse_m(params.ks, circshift(k,ii)) mse_m(params.ks, -circshift(k,ii))]);
    wk1 = ws1+ks1;
    if wk1<wk
        % display(['w shifts ' num2str(-ii) ', k0 shifts ' num2str(ii)])
        wk = wk1;
        ws=ws1;
        ks=ks1;
        w0 = circshift(w,-ii);
        if minid==1
            k0 = circshift(k,ii);
        else
            k0 = -circshift(k,ii);
        end
    end
end

for ii=0:step-1
    ws1 = min([mse_m(params.ws, circshift(w,ii)) mse_m(params.ws, -circshift(w,ii))]);
    [ks1, minid] = min([mse_m(params.ks, circshift(k,ii)) mse_m(params.ks, -circshift(k,ii))]);
    wk1 = ws1+ks1;
    if wk1<wk
        % display(['w shifts ' num2str(ii) ', k0 shifts ' num2str(ii)])
        
        wk = wk1;
        ws=ws1;
        ks=ks1;
        w0 = circshift(w,ii);
        if minid==1
            k0 = circshift(k,ii);
        else
            k0 = -circshift(k,ii);
        end
    end
end

for ii=0:step-1
    ws1 = min([mse_m(params.ws, circshift(w,-ii)) mse_m(params.ws, -circshift(w,-ii))]);
    [ks1, minid] = min([mse_m(params.ks, circshift(k,-ii)) mse_m(params.ks, -circshift(k,-ii))]);
    wk1 = ws1+ks1;
    if wk1<wk
        % display(['w shifts ' num2str(-ii) ', k0 shifts ' num2str(-ii)])
        
        wk = wk1;
        ws=ws1;
        ks=ks1;
        w0 = circshift(w,-ii);
        if minid==1
            k0 = circshift(k,-ii);
        else
            k0 = -circshift(k,-ii);
        end
    end
end

var0 = [w0;k0];
% for ii=1:length([w;k])-1
%     vv=circshift([w;k],ii);
%     w1 = vv(1:size(w,1),:);
%     k1 = vv(size(w,1)+1:end,:);
%     ws1 = min([mse_m(params.ws, w1) mse_m(params.ws, -w1)]);
%     [ks1, minid] = min([mse_m(params.ks, k1) mse_m(params.ks, -k1)]);
%     wk1 = ws1+ks1;
%     if wk1<wk
%         % display(['var shifts ' num2str(ii)])
%         wk = wk1;
%         ws=ws1;
%         ks=ks1;
%         if minid==1
%             var0 = [w1;k1];
%         else
%             var0 = [w1;-k1];
%         end
%     end
% end