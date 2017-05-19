
% This file is from pmtk3.googlecode.com

function plotKmeans(data, mu, assign, err, iter)

fprintf('iteration %d, error %5.4f\n', iter, err);
% mu
K = size(mu,2);
% figure;
symbols = {'r.', 'gx', 'bo', 'ko', 'm.'};
for k=1:K
  %subplot(2,2,iter)
  members = find(assign==k);
  plot(data(members,1), data(members, 2), symbols{k}, 'markersize', 10);
  hold on
  plot(mu(1,k), mu(2,k), sprintf('%sx', 'c'), 'markersize', 14, 'linewidth', 3)
  grid on
end
title(sprintf('iteration %d, error %5.4f', iter, err))
if iter==2, printPmtkFigure('kmeansDemoFaithfulIter2'); end

end