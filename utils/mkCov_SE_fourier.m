function [Cdiag,wvec] = mkCov_SE_fourier(rho,d,nx,nxcirc)
% [Cdiag,wvec] = mkCov_SE_fourier(rho,d,nx,nxcirc)
%
% Generates ASD ("squared exponential") covariance matrix 
% (aka "RBF kernel)" in the Fourier domain. 
%
% In real domain:
%  C_ij = rho*exp(((i-j)^2/(2*d^2)) 
%
% But in Fourier domain, is diagonal:
%  C_ii = rho*exp(-|i|^2/(2*delta^2)), delta = nxcirc/(2*pi*d)
%
% INPUTS:
%         rho - maximal prior variance ("overall scale")
%           d - length scale of ASD kernel (determines smoothness)
%          nx - number of indices (sidelength of covariance matrix)
%      nxcirc - number of coefficients to consider for circular boundary (>= nx)
%
% OUTPUT:
%   Cdiag [nxcirc x 1] - diagonal of covariance matrix
%    wvec [nxcirc x 1] - Fourier frequency for each coefficient
%
% Updated: 2015.01.11 jwp

if nxcirc < nx
    warning('mkCov_ASDfourier: nxcirc < nx. Some columns of x will be ignored');
end

% Make frequency vector
ncos = ceil((nxcirc-1)/2); % number of negative frequencies;
nsin = floor((nxcirc-1)/2); % number of positive frequencies;
wvec = [0:ncos, -nsin:-1]'; % vector of frequencies

% set delta for diagonal
delta = nxcirc/(2*pi*d); % length scale (smoothness) in Fourier domain

% compute diagonal
Cdiag = (1/rho)*normpdf(wvec,0,delta)*nxcirc; % diagonal of C matrix (without rho)
