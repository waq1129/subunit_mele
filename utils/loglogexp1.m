function [loglogef,logef,ef,iix1,iix2] = loglogexp1(x)
%  [f,df,ddf] = logexp1(x);
%
%  Computes the function:
%     f(x) = log(1+exp(x))
%  and returns first and second derivatives

ef = exp(x);
logef = log(1+ef);
loglogef = log(log(1+ef));

iix1 = logical(zeros(size(x)));
iix2 = logical(zeros(size(x)));

% Check for small values
if any(x(:)<-20)
    iix1 = (x(:)<-20);
    logef(iix1) = exp(x(iix1));
    loglogef(iix1) = x(iix1);
    
end

% Check for large values
if any(x(:)>500)
    iix2 = (x(:)>500);
    logef(iix2) = x(iix2);
    loglogef(iix2) = log(x(iix2));
    
end