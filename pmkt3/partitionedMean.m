function [M] = partitionedMean(epsilon, X, y, C)
% Group the rows of X according to the class labels in y and take the mean of each group
%
% X  - an n-by-d matrix of doubles
% y  - an n-by-1 vector of ints in 1:C
% C  - (optional) the number of classes, (calculated if not specified)
%
% M  - a C-by-d matrix of means.
% counts(i) = sum(y==i)
%
% See also partitionedSum

% This file is from pmtk3.googlecode.com


% epsilon.method = 1; % 1: DPLloyd, 2: ours
% epsilon.val = 0.9;
% epsilon.Lap = 1;
% epsilon.cdp = 0;

% if nargin < 3
% C = nunique(y);
% end
% if 1
d = size(X,2);
% N = size(X, 1);
M = zeros(C, d);
% threshold = 0.1;

%%
ndx_mat = zeros(C,1);
for c=1:C
    ndx_mat(c) = sum(y==c);
end

for c=1:C
    
    ndx = (y==c);
    if sum(ndx)==0
        rand_idx = randi(length(y), 2);
        rand_idx(rand_idx==1) = 1;
        ndx(rand_idx) = 1;
    end
    
    if epsilon.val==0
        M(c, :) = mean(X(ndx, :));
    else
        if epsilon.method==1
            
            if epsilon.cdpDPlloyd ==0
                
                num_dpts_in_c = sum(ndx);
                
                b_Lap = (d+1)*epsilon.maxiter/epsilon.val;
                noise_num_dpts_in_c = laprnd(1, 1, 0, sqrt(2)*b_Lap);
                noisedup_num_dpts_in_c = num_dpts_in_c + noise_num_dpts_in_c;
       
                
                sum_before_noise = sum(X(ndx,:));
                noise_for_sum = laprnd(1, d, 0, sqrt(2)*b_Lap);
                noisedup_sum = sum_before_noise + noise_for_sum;
                
                noisedup_mean = noisedup_sum./noisedup_num_dpts_in_c;
                
                noisedup_mean(noisedup_mean>1) = 1;
                noisedup_mean(noisedup_mean<-1)= -1;
                
                M(c, :) = noisedup_mean;
                
            else % epsilon.cdpDPlloyd ==1
                
                % this is for DPlloyd with Laplace noise and zCDP composition 
                JK = epsilon.maxiter; % laplace mechanism for J times
                myfun = @(x) (JK*x^2/2 + 2*sqrt(JK*x^2/2*log(1/epsilon.delta)) - epsilon.val)^2;
                model.eps_prime = fsolve(myfun, 0.1);
                
                
%                 mudiff = (d+1);
                b_Lap = (d+1)/model.eps_prime;
                noise_num_dpts_in_c = laprnd(1, 1, 0, sqrt(2)*b_Lap);
                
                num_dpts_in_c = sum(ndx);
                noisedup_num_dpts_in_c = num_dpts_in_c + noise_num_dpts_in_c;

                sum_before_noise = sum(X(ndx,:));
                noise_for_sum = laprnd(1, d, 0, sqrt(2)*b_Lap);
                noisedup_sum = sum_before_noise + noise_for_sum;
                
                noisedup_mean = noisedup_sum./noisedup_num_dpts_in_c;
                
                noisedup_mean(noisedup_mean>1) = 1;
                noisedup_mean(noisedup_mean<-1)= -1;
                
                M(c, :) = noisedup_mean;
            end
            
            
        else % epsilon.method ==0 (our method)
            
            M_c = mean(X(ndx, :));
            
            %             ndx_mat(ndx_mat<=1) = 1.01;
            sorted_ndx = sort(ndx_mat);
            sorted_ndx(sorted_ndx==0)=2;
            L1sen = sqrt(d)*1./sorted_ndx;
                       
            if epsilon.Lap==1
                
                JK = epsilon.maxiter; % laplace mechanism for J times
                myfun = @(x) (JK*x^2/2 + 2*sqrt(JK*x^2/2*log(1/epsilon.delta)) - epsilon.val)^2;
                model.eps_prime = fsolve(myfun, 0.1);
                b_Lap = L1sen(c)/model.eps_prime;
                noise_for_mean = laprnd(1, 2, 0, sqrt(2)*b_Lap);
                
            else
                
                mudiff = L1sen(c)/sqrt(d);
                delta = 1e-6;
                c2 = 2*log(1.25/delta);
                
                JK = epsilon.maxiter; % laplace mechanism for J times
                myfun = @(x) (JK*x^2/(2*c2) + 2*sqrt(JK*x^2/(2*c2)*log(1/epsilon.delta))*x - epsilon.val)^2;
                model.eps_prime = fsolve(myfun, 0.1);
                sigma = mudiff/model.eps_prime;
                
                noise_for_mean = mvnrnd(zeros(1,d), c2*sigma^2*eye(d));
            end
            
            noisedup_mean = M_c + noise_for_mean;
            
            noisedup_mean(noisedup_mean>1) = 1;
            noisedup_mean(noisedup_mean<-1)= -1;
            
            % noisedup_mean
            
            M(c, :) = noisedup_mean;
        end
    end
    
end

end




