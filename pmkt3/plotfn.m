function plotfn(model, X, ess, loglik, iter)
post = ess.weights; mu = model.cpd.mu; Sigma = model.cpd.Sigma;
% str = sprintf('iteration %d, loglik %5.4f\n', iter, loglik);
n = size(X, 1);
% colors = [post(:,1), zeros(n, 1), post(:,2)]; % fraction of red and blue
% figure(1); clf; 
hold on;
for i=1:n
  plot(X(i, 1), X(i, 2), 'ko', 'MarkerSize', 6);
end
% classColors = 'rb';
K  = size(mu,2);
if K==2
    classColors = 'rb';
elseif K==3
    classColors = 'rbg';
end

for k=1:K
  gaussPlot2d(mu(:,k), Sigma(:,:,k), 'color', classColors(k));
  plot(mu(1,k), mu(2,k),'o','linewidth',2, 'color', classColors(k));
end
% title(str)
axis equal
box on
pause(0.1);
%set(gca, 'YTick', -2:2);
%pause
% fname = sprintf('mixGaussDemo_%s_iter%d', dataset, iter);
% printPmtkFigure(fname);
end