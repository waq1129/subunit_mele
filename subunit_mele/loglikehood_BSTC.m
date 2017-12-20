function f = loglikehood_BSTC(C, b, a, x, y, nSpikes, opt)
%% Poisson log likelihood 

xCx = sum((x*C).*x,2)/2;
tmp0 = xCx+x*b+a;

% f = sum(y .* tmp0 - expterm*exp(max_tmp)) / nSpikes;
if strcmp(opt.nonl, 'exp')
    max_tmp = max(tmp0);
    tmp = tmp0-max_tmp;
    expterm = exp(tmp);
    f = sum(y .* tmp0 - expterm*exp(max_tmp)) / nSpikes;
end


if strcmp(opt.nonl, 'rec')
    [loglogef,logef] = loglogexp1(tmp0);
    f = sum(y.*loglogef-logef)/nSpikes;  %% change with nonl
end
