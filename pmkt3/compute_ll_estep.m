function [ess, loglik] = compute_ll_estep(model, data)
% this is just a copy of [ess, loglik] = estep(model, data)
%% Compute the expected sufficient statistics
[weights, ll] = mixGaussInferLatent(model, data); 
cpd           = model.cpd;
ess           = cpd.essFn(cpd, data, weights); 
ess.weights   = weights; % useful for plottings
loglik        = sum(ll) + cpd.logPriorFn(cpd) + model.mixPriorFn(model); 
end