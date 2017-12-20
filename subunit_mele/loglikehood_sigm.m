function [f, df, ddf] = loglikehood_sigm(V, x, y, opt, nSpikes)
% log-likelihood with sigmoid subunit nonlinearity
%% unpack variables
shift = opt.shift;
nkt = opt.nkt;
dws = opt.nDim_ws;
dks = opt.nDim_ks;
nModel = opt.nModel;
nDim = opt.nDim;
opt.nDim_k = opt.nDim_ks(1);
n = size(x,1);
B = fft(eye(nDim));
Bx = fft(x')/nDim;
wid = dks(1):nkt:nDim;
Bw = B(:,wid);
Bk = B(:,1:dks(1));

%%
r = zeros(n,1);
start_id = 1;
for mm = 1:nModel
    nDim_k = dks(mm);
    dw = dws(mm);
    end_id = start_id+nDim_k+dw-1;
    var = V(start_id:end_id);
    w = var(1:dw);
    k = var(1+dw:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    K = formK_Fourier(k, opt_mm);
    Kx = K*x';
    r = r+(1./(exp(-Kx+shift)+1))'*w;
    start_id = end_id+1;
end

%% choose output nonlinearity
if strcmp(opt.nonl, 'rec')
    [loglogef,logef,ef,iix1,iix2] = loglogexp1(r);
    f = -sum(y.*loglogef-logef)/nSpikes;  %% change with nonl
    
    ye = (y./logef-1).*(1-1./(1+ef));  %% change with nonl
    yeg = (y./logef.^2).*(1-1./(1+ef)).^2-(y./logef-1).*ef./(1+ef).^2;  %% change with nonl
    
    % exp
    ye(iix1) = y(iix1)-ef(iix1);
    yeg(iix1) = ef(iix1);
    
    % linear x
    ye(iix2) = y(iix2)./r(iix2)-1;
    yeg(iix2) = y(iix2)./r(iix2).^2;
    
end
if strcmp(opt.nonl, 'exp')
    ef = exp(r);
    ye = y-ef;
    yeg = ef;
    f = -sum(y.*r-ef)/nSpikes;
end

%% gradient and hessian
ddf = sparse(length(V), length(V));
ddf1 = sparse(length(V), length(V));
df = zeros(length(V),1);
Bxye = bsxfun(@times, Bx, ye');
tBx = transpose(Bx');
BX = opt.BX;
start_id = 1;
mid = 0;
for mm = 1:nModel
    nDim_k = dks(mm);
    dw = dws(mm);
    end_id = start_id+nDim_k+dw-1;
    var = V(start_id:end_id);
    w = var(1:dw);
    k = var(1+dw:end);
    opt_mm = opt;
    opt_mm.nDim_k0 = opt.nDim_k0s(mm);
    opt_mm.nDim_k = opt.nDim_ks(mm);
    opt_mm.nDim_w = opt.nDim_ws(mm);
    K = formK_Fourier(k, opt);
    Kx = K*x';
    eKx = exp(-Kx+shift);
    sigKx = 1./(eKx+1);
    if mm>mid
        %         display(['mm gen ' num2str(mm)])
        % Hww
        gw = sigKx;
        gwef = bsxfun(@times, gw, yeg');
        Hww = -gwef*gw';
        
        % Hkk
        sigKx1 = sigKx.^2.*eKx;
        sigKx1(isnan(sigKx1)) = 0;
        BWx = bsxfun(@times, Bw, w')*sigKx1;
        XX = tBx.*BWx;
        gk = Bk'*XX;
        gkef = bsxfun(@times, gk, yeg');
        gkgk = gkef*gk';
        
        h = sigKx.^3.*eKx.*(eKx-1);
        h(isnan(h)) = 0;
        BW = zeros(nDim^2, n);
        Bw1 = transpose(Bw*bsxfun(@times, w, h))';
        for ii=1:nDim
            BW((ii-1)*nDim+1:ii*nDim, :) = circshift(Bw1,ii-1);
        end
        BXye = bsxfun(@times, BX, ye');
        HFkk = Bk'*transpose(reshape(sum(BW.*BXye,2), nDim, []))*Bk;
        Hkk = HFkk-gkgk;
        
        % Hwk
        gwgk = gwef*gk';
        HFwk = (Bw'.*transpose(Bxye*sigKx1'))*Bk;
        Hwk = HFwk-gwgk;
        
        % collect g
        mid = mid+1;
        gws{mid} = gw;
        gks{mid} = gk;
    else
        %         display(['mm pick ' num2str(mm)])
        gw = gws{mm};
        gk = gks{mm};
        
        % Hww
        gwef = bsxfun(@times, gw, yeg');
        Hww = -gwef*gw';
        
        % Hkk
        gkef = bsxfun(@times, gk, yeg');
        h = sigKx.^3.*eKx.*(eKx-1);
        h(isnan(h)) = 0;
        BW = zeros(nDim^2, n);
        Bw1 = transpose(Bw*bsxfun(@times, w, h))';
        for ii=1:nDim
            BW((ii-1)*nDim+1:ii*nDim, :) = circshift(Bw1,ii-1);
        end
        BXye = bsxfun(@times, BX, ye');
        HFkk = Bk'*transpose(reshape(sum(BW.*BXye,2), nDim, []))*Bk;
        gkgk = gkef*gk';
        Hkk = HFkk-gkgk;
        
        % Hwk
        sigKx1 = sigKx.^2.*eKx;
        sigKx1(isnan(sigKx1)) = 0;
        gwgk = gwef*gk';
        HFwk = (Bw'.*transpose(Bxye*sigKx1'))*Bk;
        Hwk = HFwk-gwgk;
    end
    
    ddf(start_id:end_id, start_id:end_id) = real([Hww Hwk; Hwk' Hkk]);
    
    % gw, gk
    g = [gw; gk];
    gF = g*ye;
    
    df(start_id:end_id) = gF;
    
    start_id1 = end_id+1;
    
    for nn = mm+1:nModel
        nDim_k1 = dks(nn);
        dw1 = dws(nn);
        end_id1 = start_id1+nDim_k1+dw1-1;
        
        if nn>mid
            %             display(['nn gen ' num2str(nn)])
            var = V(start_id1:end_id1);
            w1 = var(1:dw1);
            k1 = var(1+dw1:end);
            opt_nn = opt;
            opt_nn.nDim_k0 = opt.nDim_k0s(nn);
            opt_nn.nDim_k = opt.nDim_ks(nn);
            opt_nn.nDim_w = opt.nDim_ws(nn);
            K1 = formK_Fourier(k1, opt_nn);
            Kx1 = K1*x';
            eKx1 = exp(-Kx1);
            sigKx1 = 1./(eKx1+1);
            
            % Hww1
            gw1 = sigKx1;
            Hww1 = -gwef*gw1';
            
            % Hkk1
            sigKx11 = sigKx1.^2.*eKx1;
            sigKx11(isnan(sigKx11)) = 0;
            BWx1 = bsxfun(@times, Bw, w1')*sigKx11;
            XX1 = tBx.*BWx1;
            gk1 = Bk'*XX1;
            gkgk1 = gkef*gk1';
            Hkk1 = -gkgk1;
            
            % Hwk
            Hwk1 = -gwef*gk1';
            Hkw1 = -gkef*gw1';
            
            % collect g
            mid = mid+1;
            gws{mid} = gw1;
            gks{mid} = gk1;
            
        else
            %             display(['nn pick ' num2str(nn)])
            gw1 = gws{nn};
            gk1 = gks{nn};
            
            % Hww1
            Hww1 = -gwef*gw1';
            
            % Hkk
            Hkk1 = -gkef*gk1';
            
            % Hwk
            Hwk1 = -gwef*gk1';
            Hkw1 = -gkef*gw1';
        end
        
        ddf1(start_id:end_id, start_id1:end_id1) = real([Hww1 Hwk1; Hkw1 Hkk1]);
        
        start_id1 = end_id1+1;
    end
    start_id = end_id+1;
end

ddf1 = ddf1+ddf1';
ddf = ddf+ddf1;

df = -real(df/nSpikes);
ddf = -real(ddf/nSpikes);

%% smoothing
if ~(~isfield(opt, 'smoothing') || opt.smoothing == 0)
    invKse = opt.invKse*opt.lambda_w;
    if opt.smoothk
        invKse1 = opt.invKse1*opt.lambda_k;
    end
    
    start_id = 1;
    for mm = 1:nModel
        nDim_w = dws(mm);
        nDim_k = dks(mm);
        
        end_id = start_id+nDim_k+nDim_w-1;
        var = V(start_id:end_id);
        w = var(1:nDim_w);
        if opt.smoothing == 2
            w = w/norm(w);
        end
        f = f + w'*invKse*w/2;
        df(start_id:start_id+nDim_w-1) = df(start_id:start_id+nDim_w-1) + invKse*w;
        ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) = ddf(start_id:start_id+nDim_w-1, start_id:start_id+nDim_w-1) + invKse;
        
        if opt.smoothk
            k = var(1+nDim_w:end);
            f = f + k'*invKse1*k/2;
            df(start_id+nDim_w:end_id) = df(start_id+nDim_w:end_id) + invKse1*k;
            ddf(start_id+nDim_w:end_id, start_id+nDim_w:end_id) = ddf(start_id+nDim_w:end_id, start_id+nDim_w:end_id) + invKse1;
        end
        
        start_id = end_id+1;
    end
end

%% clear memory
clearvars -except f df ddf
