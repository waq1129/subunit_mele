%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test script for real neuron data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear,clf,addpath(genpath(pwd));

load('544l029.p21_stc.mat');
spikes_per_frm = spikes_per_frm';
isDebugging = 0;

iStartTest = 5e4;
nshft = 3;

% The first few frames are abandoned
spikes_per_frm = spikes_per_frm(nshft+1:end);
% The last few stimuli are abandoned
stim = stim(1:end-nshft,:);

xDim = size(stim, 2); % dimension of stimuli
dTimeEmbedding = 10; % number of time bins to include in temporal kernel
nx = xDim * dTimeEmbedding; % total dimension of stimuli

%% Formulate simulus matrix from raw stim
% randn('seed', 20110522);
xall = makeStimRows(stim, dTimeEmbedding); % converts spatio-temporal stimulus to a design matrix
nsamp = size(xall,1); % total number of data points
N = 5e4; % training size
Ntest = 5e4; % test size
indstr = 1:N;
indstst = nsamp-Ntest+1:nsamp;

x = xall(indstr,:); % training data
y = spikes_per_frm(indstr);

x_test = xall(indstst,:); % test data
y_test = spikes_per_frm(indstst);

%% Subunit
% Single neuron
[loglikeli1, runtime1, params_est1, opt1] = subunit_mele(x,y,x_test,y_test,[5],xDim,dTimeEmbedding,1);

% Two neurons
[loglikeli2, runtime2, params_est2, opt2] = subunit_mele(x,y,x_test,y_test,[5;5],xDim,dTimeEmbedding,1);

