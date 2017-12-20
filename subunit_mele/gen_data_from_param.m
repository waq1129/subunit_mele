function [x_train, y_train, x_test, y_test, opt] = gen_data_from_param(params, opt)
opt.nSamples = opt.nSamples_train+opt.nSamples_test;

%% gen x and y
rho = 0;
Sigma = toeplitz(rho.^((1:opt.nDim)-1)); % sample covariance

% Generate input
% x: (opt.nSamples x nDim)
x = mvnrnd(zeros(opt.nDim, 1), Sigma, opt.nSamples);

% nonlinearity
if strcmp(opt.nonl,'exp')
    fnl = @(x) exp(x);
end
if strcmp(opt.nonl,'rec')
    fnl = @(x) logexp1(x);
end

if strcmp(opt.sub, 'quad')
    f = @(t,x) (t.a + x * t.b + .5 * ((x * t.W).^2) * t.Ds);
end

if strcmp(opt.sub, 'sigm')
    f = @(t,x) ((1./(1+exp(-x*t.K'+opt.shift)))*t.w);
end

% if opt.plotfig && strcmp(opt.sub, 'sigm') % plot sigmoid nonlinearity to inspect the output range
%     t = params.thetas{1};
%     aa = x*t.K';
%     mina = min(min(aa));
%     maxa = max(max(aa));
%     x0 = mina:maxa;
%     subplot(221),plot(x0,1./(exp(-x0+opt.shift)+1),'o'); drawnow
%     x0 = -10:10;
%     subplot(222),plot(x0,1./(exp(-x0+opt.shift)+1),'o'); drawnow
% end
if opt.plotfig % plot true w and k
    subplot(221),plot(params.ws); title('w')
    subplot(222),plot(params.ks); title('k')
end

fx = zeros(opt.nSamples,1);
for mm=1:opt.nModel
    fx = fx+f(params.thetas{mm}, x);
end
if opt.plotfig
    subplot(223),hist(fx),title('lambda'); drawnow
end
fx = fnl(fx);
if opt.plotfig
    subplot(224),plot(fx),title('exp(lambda)'); drawnow
end
fprintf('Simulating...\n');
y = poissrnd(fx);
nSpikes = sum(y);
st = find(y > 0); st = st(:); % column vector of spike indices
sps = nSpikes / opt.nSamples;
maxY = max(y);
fprintf('Total [%d] spikes, max spikes per bin [%d], average firing rate [%f] per bin\n', nSpikes, maxY, sps);

if maxY > 100
    warning('You have a HUGE number of spikes in ONE bin! This will dominate your covariance matrix!');
end

% split into train and test
x_train = x(1:opt.nSamples_train,:);
x_test = x(opt.nSamples_train+1:opt.nSamples_train+opt.nSamples_test,:);
y_train = y(1:opt.nSamples_train,:);
y_test = y(opt.nSamples_train+1:opt.nSamples_train+opt.nSamples_test,:);

opt.nSpikes = sum(y_train);
opt.nSpikes_test = sum(y_test);
