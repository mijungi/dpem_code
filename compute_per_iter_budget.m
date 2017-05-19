% test compositions with combinations of mechanisms

function model = compute_per_iter_budget(epsilon)

J = epsilon.J;
K = epsilon.K; 
D = epsilon.D;
delta_i = 1e-8;

if epsilon.comp == 0
    % non-private
    model.eps_prime = 0;
    
elseif epsilon.comp == 1
    % linear composition
    
    if epsilon.lap == 1
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
        JK = J*(2*K+1);
        myfun = @(x) (JK*x*(exp(x)-1) + sqrt(2*JK*log(1/epsilon.total_del))*x - epsilon.total_eps)^2;
        model.eps_prime = fsolve(myfun, 0.1);
    else % gaussian
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
        delta_prime = epsilon.total_del - J*(K+1)*delta_i;
        JK = J*(2*K+1);
        myfun = @(x) (JK*x*(exp(x)-1) + sqrt(2*JK*log(1/delta_prime))*x - epsilon.total_eps)^2;
        model.eps_prime = fsolve(myfun, 0.1);
    end
    
elseif epsilon.comp == 3
    % zCDP composition
    if epsilon.lap == 1
        JK = J*(2*K+1);
        myfun = @(x) (JK*(x^2)/2 + 2*sqrt(JK*(x^2)/2*log(1/epsilon.total_del)) -  epsilon.total_eps)^2;
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
        
        maxlam = 100;
%         howmanyeps = 100;
%         eps_i_mat = linspace(1e-6, 1-1e-6, howmanyeps);
%         del_val = zeros(maxlam, howmanyeps);
        eps_i_mat = zeros(maxlam,1);
        for lam=1:maxlam
%             for ep = 1:howmanyeps
                trm1 = (1+lam)/(2*lam+1);
                trm2 = lam/(2*lam+1);
%                 myfun = @(x) J*(K+1)*log(trm1*exp(lam*x) + trm2*exp(-x*(lam+1))) + J*K*lam*x  - lam*epsilon.total_eps;
%                 del_val(lam, ep) = myfun(ep);
%             end
%             myfun = @(x) (-log(epsilon.total_del) + J*(K+1)*log(trm1*exp(lam*x) + trm2*exp(-x*(lam+1))) + J*K*lam*x - lam*epsilon.total_eps)^2;
%             myfun = @(x) (-log(epsilon.total_del) + J*K*log(trm1*exp(lam*(x/K)) + trm2*exp(-(x/K)*(lam+1))) ...
%                 + J*K*D*log(trm1*exp(lam*(x/D)) + trm2*exp(-(x/D)*(lam+1))) ...
%                 + J*K*lam*x - lam*epsilon.total_eps)^2;
            myfun = @(x) (-log(epsilon.total_del) + J*K*log(trm1*exp(lam*x) + trm2*exp(-x*(lam+1))) ...
                + J*K*D*log(trm1*exp(lam*x) + trm2*exp(-x*(lam+1))) ...
                + J*K*lam*D^2*x - lam*epsilon.total_eps)^2;

            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb = 0;
            ub = 1;
            eps_i_mat(lam) = fmincon(myfun, 0.1, A, b, Aeq, beq, lb, ub);
        end
        eps_i_mat_sorted = sort(eps_i_mat, 'descend');
        model.eps_prime = eps_i_mat_sorted(1)*D^2;

    else
        c2 = 2*log(1.25/delta_i);
        model.c2 = c2; 
%         c2_pi = 2*log(1.25/(delta_i/K));
%         model.c2_pi = c2_pi; 
%         c2_mu = 2*log(1.25/(delta_i/D));
%         model.c2_mu = c2_mu; 
        
        maxlam = 100;
        eps_i_mat = zeros(maxlam,1);
        for lam=1:maxlam
            trm1 = lam^2 + lam;
%             myfun = @(x) (-log(epsilon.total_del) + J*(K+1)*trm1*(x^2)/(2*c2) + J*K*lam*x - lam*epsilon.total_eps)^2;
            myfun = @(x) (-log(epsilon.total_del) ...
                + J*K*trm1*(x^2)/(2*c2) ...
                + J*K*D*trm1*(x^2)/(2*c2) ...
                + J*K*lam*D^2*x- lam*epsilon.total_eps)^2;
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb = 0;
            ub = 1;
            eps_i_mat(lam) = fmincon(myfun, 0.1, A, b, Aeq, beq, lb, ub);
        end
        eps_i_mat_sorted = sort(eps_i_mat, 'descend');
        model.eps_prime = eps_i_mat_sorted(1)*D^2;

                
    end
    
end