% life science data from UCI data repository
% mijung wrote on October 3, 2016 for aistats submission

% inputs:

% (1) total_eps: total privacy loss (budget), e.g., 0.1
% (2) total_del: total privacy tolerance

% (3) lap: do you want to use laplace mechanism or Gaussian mechanism?
%          lap = 1, for LLG scenario
%          lap = 0, for GGG scenario

% (4) comp: composition method
%     comp = 0, for non-private
%     comp = 1, for linear
%     comp = 2, for adv
%     comp = 3, for zcdp
%     comp = 4, for ma

function testLifesci(total_eps, total_del, lap, comp)


startup;

seednum = 100;
rng(seednum, 'twister');

load('data/lifesci.csv');
lifesci = lifesci(:, 1:10);
normaliser = 1./max(sqrt((sum(lifesci.^2,2))));
X = bsxfun(@times, lifesci, normaliser);

% split data into 10 training/set sets
maxfold = 10;
Indices = crossvalind('Kfold', length(X), maxfold);

maxseed = 1; 
loglik_train = zeros(maxfold, maxseed);
loglik_test = zeros(maxfold, maxseed);

for iter = 1: maxfold
    
    Xtest = X(Indices==iter, :);
    Xtrain = X(Indices~=iter, :);
    
    fprintf([num2str(iter) ' th iteration out of ' num2str(maxfold) ' \n']);

    % generate data
    D = size(Xtest, 2);
    K = 3;

    %%
    for seednum = 1:maxseed
        
        [seednum maxseed]
        
        rng(seednum, 'twister');

        %% initialisation of parameters
        
        initParams.mu = bsxfun(@plus, lhsdesign(K,D)', -0.5*ones(D, 1)) + 0.01*randn(D,K);
        initParams.mu = bsxfun(@times, initParams.mu, normaliser);
        initParams.Sigma = repmat(eye(D),[1 1 K]);
        initParams.mixWeight = normalize(ones(1,K));
        
        % conventional EM
        maxIter = 10;
        epsilon.total_eps = total_eps;
        epsilon.total_del = total_del;
        epsilon.lap = lap;
        epsilon.comp = comp;
        
        warning('off','all')
        
        [model, loglikHist] = mixGaussFit(epsilon, Xtrain, K,   'initParams', initParams, ...
            'maxIter', maxIter, 'plotfn', [], 'verbose', false);
        
        
        %% store results
        
        loglik_train(iter, seednum) = max(loglikHist(end), loglikHist(end-1))/length(Xtrain);
        [~, logliktst] = compute_ll_estep(model, Xtest);
        loglik_test(iter, seednum) = logliktst/length(Xtest);
        
    end
    
end


%%

filename = ['matfiles/lifesci_G=lap_' num2str(lap) '_epsilon=' num2str(total_eps)  '_delta=' num2str(total_del) '_comp=' num2str(comp) '.mat'];

save(filename, 'loglik_test', 'model');

end




