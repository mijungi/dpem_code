function [mu, assign, errHist] = kmeansFit(epsilon, X, K, varargin)
% Hard cluster data using kmeans.
%
%% Inputs
% epsilon has all the quantities for DP or CDP setup
%
% X          ...   X(i, :) the ith data case
% K          ...   the number of clusters to fit
%% Outputs
%
% mu         ...   mu(:, k) is the k'th center
% assign     ...   assign(i) is the cluster to which data point i is 
%                  assigned.
%% Optional (named) Inputs
%
% 'maxIter'  ...   [100] maximum number of iterations to run
% 'thresh'   ...   [1e-3] convergence threshold 
% 'plotFn'   ...   @plotfn(X, mu, assign, err, iter) called each iteration 
% 'verbose'  ...   [false] if true, display progress each iteration 
% 'mu'       ...   initial guess for the cluster centers
%% Example
% 
% [mu, assign, errHist] = kmeansFit(randn(1000, 10), 7, 'verbose', true);
% 
%% Parse inputs

% This file is from pmtk3.googlecode.com

[maxIter, thresh, plotfn, verbose, mu] = process_options(varargin, ...
    'maxIter' , 100           , ...
    'thresh'  , 1e-3          , ... 
    'plotfn'  , @(varargin)[] , ... % default does nothing
    'verbose' , false         , ...
    'mu'      , []            );    
[N,D] = size(X);
%% Initialize
%  Initialize using K data points chosen at random
if isempty(mu)
    perm = randperm(N);
    % in the unlikely event of a tie,
    % we want to ensure the means are different.
    v = var(X);
    noise = gaussSample(zeros(1, length(v)), 0.01*diag(v), K);
    mu   = X(perm(1:K), :)' + noise';
end
%% Setup loop
iter    = 1;
errHist = zeros(maxIter, 1);
prevErr = Inf; 
prevMu = zeros(size(mu));

while true
    dist   = sqDistance(X, mu'); % dist(i, j) = sum((X(i, :)- mu(:, j)').^2)
    assign = minidx(dist, [], 2); 
    %% DP stuff comes into this function
    mu     = partitionedMean(epsilon, X, assign, K)';
    currentErr = sum(sum(dist'.*bsxfun(@eq, (1:K).', assign.')))/N;
%     currentErr
%     if isnan(currentErr)
%         display;
%     end
    
%     currentErr
%     dbstop if isnan(currentErr)
%      S = bsxfun(@eq, sparse(1:K).', assign.')'.*X;
    %% Display progress

    errHist(iter) = currentErr;
%     if plotfn
%     plotfn(X, mu, assign, currentErr, iter); 
%     end
%     if verbose, fprintf('iteration %d, err = %f\n', iter, currentErr); end
    %% Check convergence
    if convergenceTest(currentErr, prevErr, thresh)  ||  (iter >= maxIter)
        mu = prevMu;
        break
    end
    iter = iter + 1;
    prevErr = currentErr; 
    prevMu = mu; 
% %     dbstop if isnan(prevErr)
end
errHist = errHist(1:iter);
end
