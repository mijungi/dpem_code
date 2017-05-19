function [err, assign] = NICV(X, mu)

    dist   = sqDistance(X, mu'); 
    N = size(X, 1); 
    assign = minidx(dist, [], 2); 
    K = size(mu, 2); 
    err = sum(sum(dist'.*bsxfun(@eq, (1:K).', assign.')))/N;