% test DP kmeans with Gowalla dataset
% mijung edits kmeansDemo from pmkt3 package
% April 28, 2016

% clear all;
% close all;

load('data/Gowalla')
X = Gowalla;
% X = Gowalla(1:107091,:);
% X = standardizeCols(X);
normaliser = 1./sqrt(max((sum(X.^2,2))));
X = bsxfun(@times, X, normaliser);

%%

% Nmat = floor(length(X)*0.01);
Nmat = floor(length(X).*0.01);
% maxseed= 1;
maxseed = 100;
D = 2;
K = 3;
nummethod = 5;
epsilonmat = 10.^linspace(-2, -0.01, 8);
% epsilonmat = 0.01;
errmat_nonpriv = zeros(maxseed,1);
errmat_dplloyd = zeros(maxseed, length(epsilonmat));
errmat_ourlap = zeros(maxseed, length(epsilonmat));
errmat_ourgau = zeros(maxseed, length(epsilonmat));
errmat_dplloyd_cdp = zeros(maxseed, length(epsilonmat));
epsilon.delta = 1e-4; 
epsilon.delta_i = 1e-6; 

% for numSeed= 56;
for numSeed = 1:maxseed
    seednum = 2+numSeed;
    rng(seednum, 'twister');
    
    X = X(randperm(length(X)),:);
    fprintf([num2str(numSeed) ' th iteration out of ' num2str(maxseed) ' \n']);
    
    
    for numN = 1: length(Nmat)
        N = Nmat(numN);
        Xtrain = X(1:N, :);
        %%
        mu = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
        mu_init = bsxfun(@times, mu, normaliser);
        %%
        maxIter = 10;
        
        for whichmeth = 1: nummethod
            
            epsilon.maxiter = maxIter;
            if whichmeth==1 % nonpriv
                %%
                %                 whichmeth
                epsilon.val=0;
                [mu, assign, errHist] = kmeansFit(epsilon, Xtrain, K, 'plotfn', [], ...
                    'maxIter', maxIter, 'mu', mu_init);
                errmat_nonpriv(numSeed) = errHist(end);
%                 figure(1); clf;
%                 subplot(221); plotKmeans(Xtrain, mu, assign, errHist(end), maxIter);
%                 set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);
                %%
            elseif whichmeth == 2 % DPLloy
                epsilon.method = 1;
                epsilon.cdpDPlloyd = 0;
                for whicheps = 1: length(epsilonmat)
                    mu = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
                    mu_init = bsxfun(@times, mu, normaliser);
                    epsilon.val=epsilonmat(whicheps);
                    [mu, assign, errHist] = kmeansFit(epsilon, Xtrain, K, 'plotfn', [], ...
                        'maxIter', maxIter, 'mu', mu_init);
                    errmat_dplloyd(numSeed, whicheps) = errHist(end);
%                     subplot(222); plotKmeans(Xtrain, mu, assign, errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);
                end
            elseif whichmeth == 3 % ours with Laplace noise
                epsilon.method = 2;
                epsilon.Lap = 1;
                for whicheps = 1: length(epsilonmat)
                    mu = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
                    mu_init = bsxfun(@times, mu, normaliser);                    
                    epsilon.val=epsilonmat(whicheps);
                    [mu, assign, errHist] = kmeansFit(epsilon, Xtrain, K, 'plotfn', [], ...
                        'maxIter', maxIter, 'mu', mu_init);
                    errmat_ourlap(numSeed, whicheps) = errHist(end);
%                     subplot(223); plotKmeans(Xtrain, mu, assign, errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);
                end
                
            elseif whichmeth == 4 % ours with Gaussian noise
                epsilon.method = 2;
                epsilon.Lap = 0;
                for whicheps = 1: length(epsilonmat)
                    mu = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
                    mu_init = bsxfun(@times, mu, normaliser);
                    epsilon.val=epsilonmat(whicheps);
                    [mu, assign, errHist] = kmeansFit(epsilon, Xtrain, K, 'plotfn', [], ...
                        'maxIter', maxIter, 'mu', mu_init);
                    errmat_ourgau(numSeed, whicheps) = errHist(end);
%                     subplot(224); plotKmeans(Xtrain, mu, assign, errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);
                end
            else % DPLloyd with CDP
                
                epsilon.method = 1;
                epsilon.cdpDPlloyd = 1;
                for whicheps = 1: length(epsilonmat)
                    epsilon.val=epsilonmat(whicheps);
                    mu = bsxfun(@plus, lhsdesign(K,2)', [-0.5; -0.5]) + 0.01*randn(D,K);
                    mu_init = bsxfun(@times, mu, normaliser);                   
                    [mu, assign, errHist] = kmeansFit(epsilon, Xtrain, K, 'plotfn', [], ...
                        'maxIter', maxIter, 'mu', mu_init);
                    errmat_dplloyd_cdp(numSeed, whicheps) = errHist(end);
%                     subplot(222); plotKmeans(Xtrain, mu, assign, errHist(end), maxIter);
%                     set(gca, 'xlim', [-1 1], 'ylim', [-1 1]);
                end
            end
            
        end
        
        %
        
    end
    
end

%%

filename = ['matfiles/septestGowalla.mat'];
save(filename, 'errmat_ourgau', 'errmat_ourlap', 'errmat_dplloyd', 'errmat_nonpriv', 'errmat_dplloyd_cdp');

