clear all
rng(2020) % Fix the random seed

%% =================================import data============================
T = readtable('Price_History_Commonwealth_bank.xlsx') % read the CBA stock data
stock_price = T(:,{'Close'}) %extract the close price as the stock price on day t
stock_price = table2array(stock_price) %Convert to array

y_t = zeros(length(stock_price)-1, 1)
%calculate log return
parfor i = 1:(length(stock_price)-1)
    y_t(i,1) = log(stock_price(i+1))-log(stock_price(i))
end

sigma_t = zeros(length(stock_price)-1, 1)

parfor i = 2:(length(stock_price)-1)
    sigma_t_2(i,1) = var(y_t(1:(i-1)))
end

%% ===============impose stationarity condition alpha + beta <1============
psi1 = rand %0 < psi1,psi2 <1
psi2 = rand

true_alpha = psi1*psi2
true_beta =psi1*(1-psi2)
w_true = 0.001

%% ============================MCMC setting================================
%theta: w, alpha, beta
k = @(w,alpha,beta) exp(0.5*log(alpha)+9*log(1-alpha)+9*log(beta)+0.5*log(1-beta)-sum(0.5*log(sigma_t_2))-sum(0.5*y_t.^2/sigma_t_2)) % function to compute the kernel k(w,alpha,beta)
N_iter = 8000; % number of interations 
N_burnin = 2000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

dim = 3
markov_chain = zeros(N,dim); 
theta_initial = [log(w_true),log(psi1/(1-psi1)),log(psi2/(1-psi2))] % starting value
markov_chain(1,:) = theta_initial
n = 1

%% =====================Run MCMC===========================================
while n < N
    %adaptive MCMC
    if n<=1000
        Sigma = 0.1^2/dim*eye(dim); 
    else
        Sigma = 0.1^2/dim*cov(markov_chain(n-1000:n,:)); % update Sigma based on the last 1000 iterations
    end
    
    epsilon = mvnrnd(zeros(dim,1),Sigma)
    proposal = markov_chain(n,:)+epsilon
    auxiliary = log(k(proposal(1),proposal(2),proposal(3))) - log(k(markov_chain(n,1),markov_chain(n,2),markov_chain(n,3)))
    a = min(exp(auxiliary),1)
    u = rand
    if u <a
        markov_chain(n+1,:) = proposal
    else
        markov_chain(n+1,:) = markov_chain(n,:)
    end
    n = n+1
end

%% ==========================plot the beta ================================
subplot(2,2,1)
plot(exp(markov_chain(:,1)))
title('w')  

subplot(2,2,2)
plot(exp(markov_chain(:,2))/(1+exp(markov_chain(:,2))))
title('alpha')

subplot(2,2,3)
plot(exp(markov_chain(:,3))/(1+exp(markov_chain(:,3))))
title('beta')

subplot(2,2,4)
yline(1)
plot(exp(markov_chain(:,2))/(1+exp(markov_chain(:,2)))*...
    (exp(markov_chain(:,3))/(1+exp(markov_chain(:,3)))))
title('stationary constraint')

%% ============================Calculate posterior ========================
a = ones(N - N_burnin,1)
%posterior mean and variance estimate
pme_w = mean(markov_chain(N_burnin+1:N,1))% posterior mean for transformed w
pve_w = mean((markov_chain(N_burnin+1:N,1) - a*pme_w).^2)%posterior variance for transformed w

pme_alpha = mean(markov_chain(N_burnin+1:N,2))% posterior mean for transformed alpha
pve_alpha = mean((markov_chain(N_burnin+1:N,2) - a*pme_alpha).^2)%posterior variance for transformed alpha

pme_beta = mean(markov_chain(N_burnin+1:N,3))% posterior mean for transformed beta
pve_beta = mean((markov_chain(N_burnin+1:N,3) - a*pme_beta).^2)%posterior variance for transformed beta

%%  =================Convert transformed parameter to true parameter=======
psi_one = exp(pme_alpha)/(1+exp(pme_alpha))
psi_two = exp(pme_beta)/(1+exp(pme_beta))

w_final = exp(pme_w)
alpha_final = psi_one*psi_two
beta_final = psi_one*(1-psi_two)

y_911 = y_t(length(y_t)) %stock return on 11 Sep 2020
sigma_911_2 = sigma_t_2(length(sigma_t_2)) %volatility on 11 Sep 2020

posterior_mean_estimate = [w_final,alpha_final, beta_final]
predictive_volatility_square = w_final + alpha_final*(y_911^2) + beta_final*(sigma_911_2)%calculate predictive_mean 