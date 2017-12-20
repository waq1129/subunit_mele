function [f, df, ddf, C, b] = loglikehood_quad(V, x, y, opt, nSpikes)
% loglikehood with quad subunit nonlinearity
%% unpack variables
nkt = opt.nkt;
dws = opt.nDim_ws;
dks = opt.nDim_ks;
nModel = opt.nModel;
nDim = opt.nDim;
opt.nDim_k = opt.nDim_ks(1);
B = fft(eye(nDim));
Bx = fft(x')/nDim;
wid = dks(1):nkt:nDim;
Bw = B(:,wid);
Bk = B(:,1:dks(1));

%%
C = zeros(nDim,nDim);
b = zeros(nDim,1);
a = V(end);
V = V(1:end-1);
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
    KWK = formC_Fourier(w, k, opt_mm);
    C = C+KWK;
    K = formK_Fourier(k, opt_mm);
    b = b+K'*w;
    start_id = end_id+1;
end
r = sum((x*C).*x,2)/2+x*b+a;

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
Hawk = zeros(length(V),1);
Bxye = bsxfun(@times, Bx, ye');
Bxyevec = sum(Bxye,2);
BxyeBx = Bxye*Bx';
tBx = transpose(Bx');
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
    
    if mm>mid
        %         display(['mm gen ' num2str(mm)])
        % Hww
        gw = Kx.^2/2+Kx;
        gwef = bsxfun(@times, gw, yeg');
        Hww = -gwef*gw';
        
        % Hkk
        BwB = bsxfun(@times, Bw, w')*Bw';
        BK = Bk*k;
        BWKx = bsxfun(@times, BwB, transpose(BK))*Bx;
        XX = tBx.*BWKx;
        gk = Bk'*XX+bsxfun(@times, Bk', transpose(Bw*w))*tBx;
        gkef = bsxfun(@times, gk, yeg');
        gkgk = gkef*gk';
        HFkk = Bk'*(BwB.*transpose(BxyeBx))*Bk;
        Hkk = HFkk-gkgk;
        
        % Hwk
        gwgk = gwef*gk';
        BdxyeB = bsxfun(@times, Bw', transpose(Bxyevec))*Bk;
        HFwk = (Bw'.*transpose(Bxye*Kx'))*Bk+BdxyeB;
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
        BwB = bsxfun(@times, Bw, w')*Bw';
        gkef = bsxfun(@times, gk, yeg');
        gkgk = gkef*gk';
        HFkk = Bk'*(BwB.*transpose(BxyeBx))*Bk;
        Hkk = HFkk-gkgk;
        
        % Hwk
        gwgk = gwef*gk';
        BdxyeB = bsxfun(@times, Bw', transpose(Bxyevec))*Bk;
        HFwk = (Bw'.*transpose(Bxye*Kx'))*Bk+BdxyeB;
        Hwk = HFwk-gwgk;
    end
    
    ddf(start_id:end_id, start_id:end_id) = real([Hww Hwk; Hwk' Hkk]);
    
    % gw, gk
    g = [gw; gk];
    gF = g*ye;
    
    df(start_id:end_id) = gF;
    
    % collect a
    Hawk(start_id:end_id) = -g*yeg;
    
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
            
            % Hww1
            gw1 = Kx1.^2/2+Kx1;
            Hww1 = -gwef*gw1';
            
            % Hkk
            BwB1 = bsxfun(@times, Bw, w1')*Bw';
            BK1 = Bk*k1;
            BWKx1 = bsxfun(@times, BwB1, transpose(BK1))*Bx;
            XX1 = tBx.*BWKx1;
            gk1 = Bk'*XX1+bsxfun(@times, Bk', transpose(Bw*w1))*tBx;
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

%% gradient and hessian for a
ga = sum(ye);
Haa = -sum(yeg);

df = -real([df; ga]/nSpikes);
ddf = -real([ddf Hawk; Hawk' Haa]/nSpikes);
df(isnan(df)) = 0;
ddf(isnan(ddf)) = 0;

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
clearvars -except f df ddf C b

