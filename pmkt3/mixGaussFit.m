function [model, loglikHist] = mixGaussFit(epsilon, data, nmix,  varargin)
%% Fit a mixture of Gaussians via MLE/MAP (using EM)
%
%
%% Inputs
% epsilon - 0 if this is non-DP. If non-zero, this is DPEM.
% data     - data(i, :) is the ith case, i.e. data is of size n-by-d
% nmix     - the number of mixture components to use
%
% This file is from pmtk3.googlecode.com


[initParams, prior, mixPrior, EMargs] = ...
    process_options(varargin, ...
    'initParams'        , [], ...
    'prior'             , [], ...
    'mixPrior'          , []);
[n, d]      = size(data);
model.type  = 'mixGauss';
model.nmix  = nmix;
model.d     = d;
model       = setMixPrior(model, mixPrior);
model.maxIter = EMargs{2};
model.N = n;

J = model.maxIter;
K = nmix;

model.total_eps = epsilon.total_eps;
model.lap = epsilon.lap;
model.comp = epsilon.comp; 

%         epsilon.total_eps = total_eps;
%         epsilon.total_del = total_del;
%         epsilon.Lap = lap;
%         epsilon.comp = comp;
 
delta_i = 1e-6; % this is for advanced, zCDP, and MA (per iteration tolerance)
% test compositions with combinations of mechanisms

if epsilon.comp == 0
    % non-private
    model.eps_prime = 0;
    
elseif epsilon.comp == 1
    % linear composition
    
    if epsilon.lap == 1
        delta_i = epsilon.total_del/(J*K);
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        model.eps_prime = epsilon.total_eps/(J*(2*K+1));
    else % Gaussian
        delta_i = epsilon.total_del/(J*(K+1));
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        model.eps_prime = epsilon.total_eps/(J*(2*K+1));
    end
    
elseif epsilon.comp == 2
    % advanced composition
    
    if epsilon.lap == 1
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        delta_prime = epsilon.total_del - J*K*delta_i;
        JK = J*(2*K+1);
        myfun = @(x) (JK*x*(exp(x)-1) + sqrt(2*JK*log(1/delta_prime))*x - epsilon.total_eps)^2;
        model.eps_prime = real(fsolve(myfun, 0.1));
    else % gaussian
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        delta_prime = epsilon.total_del - J*(K+1)*delta_i;
        JK = J*(2*K+1);
        myfun = @(x) (JK*x*(exp(x)-1) + sqrt(2*JK*log(1/delta_prime))*x - epsilon.total_eps)^2;
        model.eps_prime = real(fsolve(myfun, 0.1));
    end
    
elseif epsilon.comp == 3
    % zCDP composition
    if epsilon.lap == 1
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        myfun = @(x) (J*(K+1)*(x^2)/2 + 2*sqrt((J*(K+1)*(x^2)/2)*log(1/epsilon.total_del)) -  epsilon.total_eps)^2;
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = 0;
        ub = 1;
        model.eps_prime = fmincon(myfun, 0.1, A, b, Aeq, beq, lb, ub);
    else
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        myfun = @(x) (J*(x^2)/(2*c2) + J*K*(x^2)/(2*c2) + J*K*(x^2)/2 +  2*sqrt((J*(x^2)/(2*c2) + J*K*(x^2)/(2*c2) + J*K*(x^2)/2)*log(1/epsilon.total_del))  - epsilon.total_eps)^2;
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = 0;
        ub = 1;
        model.eps_prime = fmincon(myfun, 0.1, A, b, Aeq, beq, lb, ub);
    end
    
else % MA composition
    if epsilon.lap == 1
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
           
        maxlam = 100;
        eps_i_mat = linspace(0.001, 0.999, 200);
        logdel = zeros(maxlam, length(eps_i_mat));

        for lam=1:maxlam
            
            trm1_L = (1+lam)/(2*lam+1);
            trm2_L = lam/(2*lam+1);
            trm1_G = lam^2 + lam;
            
            myfun = @(x) J*(K+1)*(log(trm1_L + trm2_L*exp(-(2*lam+1)*x)) + lam*x) ...
                + J*K*trm1_G*(x^2)/(2*c2)- lam*epsilon.total_eps;
            
            for ep = 1:length(eps_i_mat)
                eps = eps_i_mat(ep);
                logdel(lam, ep) = myfun(eps);
            end
        end
        
        % Find the largest epsilon_i that satisfies delta constraint
        opt_del = zeros(length(eps_i_mat), 1);
        for i=1:length(eps_i_mat);
            idx = find(logdel(:,i)<log(epsilon.total_del),1);
            if size(idx,1)==0
                opt_del(i)=0;
            else
                opt_del(i) = exp(logdel(idx,i));
            end
        end
        idx_sel = find(opt_del==0, 1) - 1;
        if idx_sel==0
            model.eps_prime = eps_i_mat(1);
            model.del = opt_del(1);
        else
            model.eps_prime = eps_i_mat(idx_sel);
            model.del = opt_del(idx_sel);
        end

    else
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        
        maxlam = 100;
        eps_i_mat = linspace(0.001, 0.999, 200);
        logdel = zeros(maxlam, length(eps_i_mat));
%         eps_i_mat = zeros(maxlam,1);
        for lam=1:maxlam
            trm1 = lam^2 + lam;
            myfun = @(x) J*(2*K+1)*trm1*(x^2)/(2*c2) - lam*epsilon.total_eps;
            
            for ep = 1:length(eps_i_mat)
                eps = eps_i_mat(ep);
                logdel(lam, ep) = myfun(eps);
            end

        end
        
        % Find the largest epsilon_i that satisfies delta constraint
        opt_del = zeros(length(eps_i_mat), 1);
        for i=1:length(eps_i_mat);
            idx = find(logdel(:,i)<log(epsilon.total_del),1);
            if size(idx,1)==0
                opt_del(i)=0;
            else
                opt_del(i) = exp(logdel(idx,i));
            end
        end
        idx_sel = find(opt_del==0, 1) - 1; 
        model.eps_prime = eps_i_mat(idx_sel);
        model.del = opt_del(idx_sel); 
    end
end


initFn = @(m, X, r)initGauss(m, X, r, initParams, prior);
[model, loglikHist] = emAlgo(model, data, initFn, @estep, @mstep , ...
    'verbose', true, EMargs{:});
end

function model = initGauss(model, X, restartNum, initParams, prior)
%% Initialize
nmix = model.nmix;
if restartNum == 1
    if ~isempty(initParams)
        mu              = initParams.mu;
        Sigma           = initParams.Sigma;
        model.mixWeight = initParams.mixWeight;
    else
        [mu, Sigma, model.mixWeight] = kmeansInitMixGauss(X, nmix);
    end
else
    mu              = randn(d, nmix);
    regularizer     = 2;
    Sigma           = stackedRandpd(d, nmix, regularizer);
    model.mixWeight = normalize(rand(1, nmix) + regularizer);
end
model.cpd = condGaussCpdCreate(mu, Sigma, 'prior', prior);
end


function [ess, loglik] = estep(model, data)
%% Compute the expected sufficient statistics
[weights, ll] = mixGaussInferLatent(model, data);
cpd           = model.cpd;
ess           = cpd.essFn(cpd, data, weights);
ess.weights   = weights; % useful for plottings
loglik        = sum(ll) + cpd.logPriorFn(cpd) + model.mixPriorFn(model);
end

function model = mstep(model, ess)
%% Maximize
cpd             = model.cpd;
model.cpd       = cpd.fitFnEss(cpd, ess, model);
model.mixWeight = normalize(ess.wsum + model.mixPrior - 1);

if model.total_eps~=0
    N = model.N;
    
    if model.lap==1
        laplace_noise_variance = 2/(N*model.eps_prime);
        noise = laprnd(model.nmix, 1, 0, sqrt(2)*laplace_noise_variance);
        noisedup_mixWeight = model.mixWeight + noise';
        
    else
        sensitiv = 2/N;
        c2 = model.c2;
        d = model.nmix; 
        sigma = sensitiv/model.eps_prime;
        noise = mvnrnd(zeros(1,d), c2*sigma^2*eye(d))';
        noisedup_mixWeight = model.mixWeight + noise';
    end
    
%     noisedup_mixWeight = model.mixWeight + noise';
    noisedup_mixWeight(noisedup_mixWeight<0) = 0;
    noisedup_mixWeight(noisedup_mixWeight>1) = 1;
    noisedup_numerator = normalize(noisedup_mixWeight).*N;
    [model.mixWeight]= normalize(noisedup_numerator + model.mixPrior - 1);
    
else
    [model.mixWeight]= normalize(ess.wsum + model.mixPrior - 1);
end

end


