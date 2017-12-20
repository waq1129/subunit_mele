function [f,ef,iix1,iix2] = logexp1(x)
%  [f,df,ddf] = logexp1(x);
%
%  Computes the function:
%     f(x) = log(1+exp(x))
%  and returns first and second derivatives

ef = exp(x);
f = log(1+ef);
iix1 = zeros(size(x));
iix2 = zeros(size(x));

% Check for small values
if any(x(:)<-20)
    iix1 = (x(:)<-20);
    f(iix1) = exp(x(iix1));
    df(iix1) = f(iix1);
    ddf(iix1) = f(iix1);
end

% Check for large values
if any(x(:)>500)
    iix2 = (x(:)>500);
    f(iix2) = x(iix2);
    df(iix2) = 1;
    ddf(iix2) = 0;
end