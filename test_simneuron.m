%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test script for a simulated complex cell
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear,clf,addpath(genpath(pwd));

neuron = load('simulated_complex_cell.mat');
dTimeEmbedding = neuron.dTimeEmbedding; % number of time bins to include in temporal kernel
xDim = neuron.xDim; % dimension of stimuli
nx = xDim * dTimeEmbedding; % total dimension of stimuli
xall = neuron.x; % stimulus design matrix
spikes_per_frm = neuron.y; % spike train

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

