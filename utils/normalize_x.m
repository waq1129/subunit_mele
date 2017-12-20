function x = normalize_x(x)
x = x-repmat(mean(x,1),size(x,1),1);
x = x./repmat(sqrt(sum(x.^2,1)),size(x,1),1);
