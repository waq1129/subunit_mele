function opt = cv_smooth(x, y, opt)
% use cross validation to find optimal hyperparameters for smoothing
% kernels for w and k. For each set of hyperparameters, use BSTC to find
% a solution for w and k first, then use the square loss objective
% function combined with smoothing priors for w and k to find smooth
% w and k. The log likelihood on the test set will be evaluated to find
% optimal hyperparameters. What we care about in this part is the set of
% optimal hyperparameters for smoothing w and k.

display('cross validating for smoothing priors')

% 5-fold cross validation
k = 5;
c = cvpartition(size(y,1),'kfold',k);
ind = randperm(size(y,1));
x = x(ind,:);
y = y(ind);

% rhow = [1e-1,1,5];
% deltaw = [1,3,5];
% rhok = [1e-1,1,5];
% deltak = [1,3,5];
rhow = [1];
deltaw = [3];
rhok = [1];
deltak = [3];
lambda_w = [1e-3 1e-2 1e-1 1];
lambda_k = [1e-5 1e-4 1e-3 1e-2];

if opt.smoothk
    [rw, dw, lw, rk, dk, lk] = ndgrid(rhow, deltaw, lambda_w, rhok, deltak, lambda_k);
    theta0 = [rw(:) dw(:) lw(:) rk(:) dk(:) lk(:)];
else
    [rw, dw, lw] = ndgrid(rhow, deltaw, lambda_w);
    theta0 = [rw(:) dw(:) lw(:)];
end
loglikelils = zeros(size(theta0,1),k);

for i=1:k
    display(['validating set ' num2str(i)])
    xtrain = x(c.training(i),:);
    ytrain = y(c.training(i),:);
    xtest = x(c.test(i),:);
    ytest = y(c.test(i),:);
    
    nSpikes = sum(ytrain);
    nSpikes_test = sum(ytest);
    nSamples = size(xtrain,1);
    
    % Estimate STC
    fprintf('Computing STC...\n');
    [sta, STC, rawmu, rawcov] = simpleSTC(xtrain, ytrain, 1); % train
    
    % MELE
    Phi = rawcov;
    [Phi1, invPhi] = svd_inv(Phi, 1e-4);
    [STC1, invSTC] = svd_inv(STC, 1e-4);
    b_bstc = invSTC*sta';
    C_bstc = invPhi - invSTC;
    [uu, ss, vv]=svd(C_bstc);
    ds = diag(ss);
    cds = cumsum(ds);
    id = cds<cds(end)*0.8;
    C_bstc = uu(:,id)*ss(id,id)*vv(:,id)';
    a_bstc = log(nSpikes/nSamples*det(invSTC*Phi)^(0.5)*exp(-sta*invPhi*invSTC*sta'/2));
    
    for ii=1:size(theta0,1)
        opt1 = opt;
        opt1.smoothing = 1;
        opt1.rho = theta0(ii,1);
        opt1.d = theta0(ii,2);
        opt1.lambda_w = theta0(ii,3);
        opt1.smoothk = opt.smoothk;
        if opt1.smoothk
            opt1.rho1 = theta0(ii,4);
            opt1.d1 = theta0(ii,5);
            opt1.lambda_k = theta0(ii,6);
        end
        opt1 = setopt_smooth(opt1);
        
        var_init = subunit_init_moment(C_bstc, b_bstc, a_bstc, x, y, opt1, 1); %initialize var
        [var_ls, w_ls, k_ls, C_ls, b_ls, a_ls] = msl_ms_wrap(var_init, C_bstc, b_bstc, Phi, invPhi, opt1); % MSL
        floss_test_BSTC = @(C, b, a) loglikehood_BSTC(C, b, a, xtest, ytest, nSpikes_test, opt1); %likelihood for C_bstc
        loglikelils(ii,i) = floss_test_BSTC(C_ls, b_ls, a_ls);
    end
end
loglikelils = sum(loglikelils,2);
% plot(loglikelils)
[maxvalue, maxid] = max(loglikelils);
theta = theta0(maxid,:);

% the set of optimal hyperparameters for smoothing priors
opt.smoothing = 1;
opt.rho = theta(1);
opt.d = theta(2);
opt.lambda_w = theta(3);
if opt.smoothk
    opt.rho1 = theta(4);
    opt.d1 = theta(5);
    opt.lambda_k = theta(6);
end
opt.loglikelils = loglikelils;










