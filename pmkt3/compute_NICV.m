
clear;
clc;


seednum = 100; 
rng(seednum, 'twister');

% load data
load('data/Gowalla')
X = Gowalla;
normaliser = 1./sqrt(max((sum(X.^2,2))));
X = bsxfun(@times, X, normaliser);
maxIter = 40; 

% for EM 
% filename = ['matfiles/gowalla_G=lap_' num2str(0) '_epsilon=' num2str(0)  '_delta=' num2str(0) '_comp=' num2str(0) '.mat'];
% load(filename);
% 
% [err_EM, assign] = NICV(X, model.cpd.mu); 
% 
% 
N = size(X,1);
rpidx = randperm(N);
rpidx = rpidx(1:7000);
% figure(1);
% plotKmeans(X(rpidx,:), model.cpd.mu, assign(rpidx), err_EM, maxIter);


K = 5;
D = 2; 

epsilon.Lap = 1;
epsilon.val = 0; 
epsilon.method = 0;
epsilon.maxiter = maxIter; 
% epsilon.delta_i = 1e-6; 
epsilon.delta = 1e-4; 

mu_init = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.05*randn(D,K);
mu_init = mu_init/normaliser;
[mu, assign, errHist] = kmeansFit(epsilon, X, K, 'plotfn', [], ...
    'maxIter', maxIter, 'mu', mu_init);

figure(1);
plotKmeans(X(rpidx,:), mu, assign(rpidx), errHist(end), maxIter);

%%

% ep_val = 0.01; 
% epsilon.val = ep_val; 
% % % for DPEM with zCDP
% 
% % the full EM % 
% initParams.mu = bsxfun(@plus, lhsdesign(K,D)', -0.5*ones(D, 1)) + 0.01*randn(D,K);
% initParams.mu = bsxfun(@times, initParams.mu, normaliser);
% initParams.Sigma = repmat(eye(D),[1 1 K]);
% initParams.mixWeight = normalize(ones(1,K));
% 
% % conventional EM
% epsilon.total_eps = ep_val;
% epsilon.total_del = 1e-4;
% epsilon.lap = 1;
% epsilon.comp = 3;
% 
% [model, loglikHist] = mixGaussFit(epsilon, X, K,   'initParams', initParams, ...
%     'maxIter', maxIter, 'plotfn', [], 'verbose', false);

%%

% [err_EM, assign] = NICV(X, model.cpd.mu);
% 
% figure(2);
% plotKmeans(X(rpidx,:), model.cpd.mu, assign(rpidx), err_EM, maxIter);

ep_val = 0.01; 
epsilon.Lap = 1;
epsilon.val = ep_val; 
epsilon.method = 0;
epsilon.maxiter = maxIter; 
% epsilon.delta_i = 1e-6; 
epsilon.delta = 1e-4; 

% mu_init = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
[mu, assign, errHist] = kmeansFit(epsilon, X, K, 'plotfn', [], ...
    'maxIter', maxIter, 'mu', mu_init);

figure(2);
plotKmeans(X(rpidx,:), mu, assign(rpidx), errHist(end), maxIter);


% 
% epsilon.Lap = 1;
% epsilon.val = ep_val; 
% epsilon.method = 0;
% epsilon.maxiter = maxIter; 
% % epsilon.delta_i = 1e-6; 
% epsilon.delta = 1e-4; 
% 
% mu_init = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
% [mu, assign, errHist] = kmeansFit(epsilon, X, K, 'plotfn', [], ...
%     'maxIter', maxIter, 'mu', mu_init);
% 
% figure(3);
% plotKmeans(X(rpidx,:), mu, assign(rpidx), errHist(end), maxIter);

%% DPllyod

epsilon.cdpDPlloyd = 1;
epsilon.val = ep_val; 
epsilon.method = 1;
epsilon.maxiter = maxIter; 
% epsilon.delta_i = 1e-6; 
epsilon.delta = 1e-4; 

% mu_init = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
[mu, assign, errHist] = kmeansFit(epsilon, X, K, 'plotfn', [], ...
    'maxIter', maxIter, 'mu', mu_init);

figure(4);
plotKmeans(X(rpidx,:), mu, assign(rpidx), errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);


%

epsilon.cdpDPlloyd = 0;
epsilon.val = ep_val; 
epsilon.method = 1;
epsilon.maxiter = maxIter; 
% epsilon.delta_i = 1e-6; 
epsilon.delta = 1e-4; 

% mu_init = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
[mu, assign, errHist] = kmeansFit(epsilon, X, K, 'plotfn', [], ...
    'maxIter', maxIter, 'mu', mu_init);

figure(5);
plotKmeans(X(rpidx,:), mu, assign(rpidx), errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);