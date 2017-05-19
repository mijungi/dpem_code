function s = logsumexp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Michael Chen (sth4nth@gmail.com).
if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end

% function s = logsumexp(a, dim)
% % Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
% % Default is dim = 1 (columns).
% % logsumexp(a, 2) will sum across rows instead of columns.
% % Unlike matlab's "sum", it will not switch the summing direction
% % if you provide a row vector.
% 
% % Written by Tom Minka
% % (c) Microsoft Corporation. All rights reserved.
% 
% if nargin < 2
%   dim = 1;
% end
% 
% % subtract the largest in each column
% [y, i] = max(a,[],dim);
% dims = ones(1,ndims(a));
% dims(dim) = size(a,dim);
% a = a - repmat(y, dims);
% s = y + log(sum(exp(a),dim));
% i = find(~isfinite(y));
% if ~isempty(i)
%   s(i) = y(i);
% end