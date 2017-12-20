function mm = mse_m(x,y)
dd = normalize_x(x)-normalize_x(y);
mm = sum(sqrt(sum(dd.^2,1)));