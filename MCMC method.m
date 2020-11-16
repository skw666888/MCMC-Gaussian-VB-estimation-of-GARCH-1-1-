clear all
rng(2020) % Fix the random seed

%% =================================import data============================
T = readtable('Price_History_Commonwealth_bank.xlsx') % read the CBA stock data
stock_price = T(:,{'Close'}) %extract the close price as the stock price on day t
stock_price = table2array(stock_price) %Convert to array

y_t = zeros(length(stock_price)-1, 1)

parfor i = 1:(length(stock_price)-1)
    y_t(i,1) = log(stock_price(i+1)/stock_price(i))
end
y_bar = mean(y_t)

sigma_t = zeros(length(stock_price)-1, 1)

parfor i = 1:(length(stock_price)-1)
    sigma_t_2(i,1) = var(y_t(i))
end
%% ===============impose stationarity condition alpha + beta <1============
psi1 = rand %0 < psi1,psi2 <1
psi2 = rand

true_alpha = psi1*psi2
true_beta =psi1*(1-psi2)
w_true = 1.1

alpha = log(psi1/(1-psi1))
beta = log(psi2/(1-psi2))
w = log(w_true)
%% ============================MCMC setting================================
%theta: w, alpha, beta

k = @(w,alpha,beta) exp(0.5*log(alpha)+9*log(1-alpha)+9*log(beta)+0.5*log(1-beta)-sum(0.5*log(sigma_t_2))-sum(0.5*y_t.^2/sigma_t_2)) % function to compute the kernel k(w,alpha,beta)
N_iter = 10000; % number of interations 
N_burnin = 2000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

dim = 3
markov_chain = zeros(N,dim); 
theta_initial = randn(1,dim) % starting value
markov_chain(1,:) = theta_initial
n = 1

%% =====================Run MCMC===========================================
while n < N
    %adaptive MCMC
    if n<=2000
        Sigma = 0.1^2/dim*eye(dim); 
    else
        Sigma = 0.01^2/dim*(2000/n)*eye(dim)
        %Sigma = 2.38^2/dim*cov(markov_chain(n-1000:n,:)); % update Sigma based on the last 1000 iterations
    end
    
    epsilon = mvnrnd(zeros(dim,1),Sigma)
    proposal = markov_chain(n,:)+epsilon
    a = min(k(proposal(1),proposal(2),proposal(3))/k(markov_chain(n,1),markov_chain(n,2),markov_chain(n,3)),1)
    u = rand
    if u <a
        markov_chain(n+1, :) = proposal
    else
        markov_chain(n+1,:) = markov_chain(n,:)
    end
    n = n+1
end

%% ==========================plot the beta ================================
subplot(2,2,1)
plot(markov_chain(:,1))
title('transformed w')  

subplot(2,2,2)
plot(markov_chain(:,2))
title('transformed alpha')

subplot(2,2,3)
plot(markov_chain(:,3))
title('transformed beta')

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
w_final = exp(pme_w)
alpha_final = exp(pme_alpha)/(1+exp(pme_alpha))
beta_final = exp(pme_beta)/(1+exp(pme_beta))

y_911 = y_t(length(y_t)) %stock on 11 Sep 2020
sigma_911_2 = sigma_t(length(sigma_t))

posterior_mean_estimate = [w_final,alpha_final, beta_final]

predictive_vol_square = w_final + alpha_final*(y_911^2) + beta_final*(sigma_911_2)%calculate predictive_mean 
