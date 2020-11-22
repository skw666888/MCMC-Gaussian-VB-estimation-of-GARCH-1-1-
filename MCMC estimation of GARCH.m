clear all;
rng(2020); % Fix the random seed

%% =================================import data============================
data = readtable('Price_History_Commonwealth_bank.xlsx'); % read the CBA stock data
stock_price = data(:,{'Close'}); %extract the close price as the stock price on day t
stock_price = table2array(stock_price); %Convert to array

y = zeros(length(stock_price)-1, 1);
%calculate log return
parfor i = 1:(length(stock_price)-1)
    y(i) = log(stock_price(i+1))-log(stock_price(i));
end
T = length(y);
y_2 = y.^2;

%% ===============impose stationarity condition alpha + beta <1============
psi1 = rand ;%0<psi1,psi2<1
psi2 = rand;
w_true = rand;
theta1 = log(psi1/(1-psi1));
theta2 = log(psi2/(1-psi2));
theta3 = log(w_true);
theta_initial = [theta1,theta2,theta3];

%% ============================MCMC setting===============================
N_iter = 100000; % number of interations 
N_burnin = 20000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

dim = 3;
markov_chain = zeros(N,dim); 
markov_chain(1,:) = theta_initial;
n = 1;

%% =====================Run MCMC===========================================
while n < N
    Sigma = 0.01*eye(dim);
    epsilon = mvnrnd(zeros(dim,1),Sigma);
    proposal = markov_chain(n,:)+epsilon;
    
    %cal k(proposal)
    alpha_p = (exp(proposal(1))/(1+exp(proposal(1))))*(exp(proposal(2))/(1+exp(proposal(2))));
    beta_p = (exp(proposal(1))/(1+exp(proposal(1))))*(1-(exp(proposal(2))/(1+exp(proposal(2)))));
    w_p = exp(proposal(3));
    
    sigma_t_2_p = zeros(T,1);
    sigma_t_2_p(1) = var(y);
    for i = 2:T
        sigma_t_2_p(i) = w_p + alpha_p*y_2(i-1) + beta_p*sigma_t_2_p(i-1);
    end
    
    log_likelihood_p = sum(-0.5*log(sigma_t_2_p)-0.5*y_2./sigma_t_2_p);
    
    %cal k(markov)
    alpha_m = (exp(markov_chain(n,1))/(1+exp(markov_chain(n,1))))*(exp(markov_chain(n,2))/(1+exp(markov_chain(n,2))));
    beta_m = (exp(markov_chain(n,1))/(1+exp(markov_chain(n,1))))*(1-(exp(markov_chain(n,2))/(1+exp(markov_chain(n,2)))));
    w_m = exp(markov_chain(n,3));
    
    sigma_t_2_m = zeros(T,1);
    sigma_t_2_m(1) = var(y);
    for i = 2:T
        sigma_t_2_m(i) = w_m + alpha_m*y_2(i-1)+ beta_m*sigma_t_2_m(i-1);
    end
    
    log_likelihood_m = sum(-0.5*log(sigma_t_2_m)-0.5*y_2./sigma_t_2_m);
     
    auxiliary=log_likelihood_p+9.5*proposal(1)+0.5*proposal(2)+...
        9*log(1+exp(proposal(1))+exp(proposal(2)))+...
        0.5*log(1+exp(proposal(2))+exp(proposal(1)+proposal(2)))-...
        19*log((1+exp(proposal(1)))*(1+exp(proposal(2))))-...
        (log_likelihood_m+9.5*markov_chain(n,1)+0.5*markov_chain(n,2)+...
        9*log(1+exp(markov_chain(n,1))+exp(markov_chain(n,2)))+...
        0.5*log(1+exp(markov_chain(n,2))+exp(markov_chain(n,1)+markov_chain(n,2)))-...
        19*log((1+exp(markov_chain(n,1)))*(1+exp(markov_chain(n,2)))));
    
    alpha = min(exp(auxiliary),1);
    u = rand;
    if u <alpha
        markov_chain(n+1,:) = proposal;
    else
        markov_chain(n+1,:) = markov_chain(n,:);
    end
    n = n+1;
end

%% ==========================plot the beta ================================
subplot(3,2,1);
plot(markov_chain(:,1));
title('Theta1');

subplot(3,2,3);
plot(markov_chain(:,2));
title('Theta2');

subplot(3,2,5);
plot(markov_chain(:,3));
title('Theta3');

subplot(3,2,2);
plot(exp(markov_chain(N_burnin+1:N,1)).*exp(markov_chain(N_burnin+1:N,2))./((1+exp(markov_chain(N_burnin+1:N,1))).*(1+exp(markov_chain(N_burnin+1:N,2)))));
title('alpha (after burn-in)');

subplot(3,2,4);
plot(exp(markov_chain(N_burnin+1:N,1))./((1+exp(markov_chain(N_burnin+1:N,1))).*(1+exp(markov_chain(N_burnin+1:N,2)))));
title('beta (after burn-in)');

subplot(3,2,6);
plot(exp(markov_chain(N_burnin+1:N,3)));
title('w (after burn-in)');  
%% ============================Calculate posterior ========================
a = ones(N - N_burnin,1);
%posterior mean and variance estimate
pme_alpha = mean(exp(markov_chain(N_burnin+1:N,1)).*exp(markov_chain(N_burnin+1:N,2))./...
    ((1+exp(markov_chain(N_burnin+1:N,1))).*(1+exp(markov_chain(N_burnin+1:N,2)))));% posterior mean for alpha
psd_alpha = std(exp(markov_chain(N_burnin+1:N,1)).*exp(markov_chain(N_burnin+1:N,2))./...
    ((1+exp(markov_chain(N_burnin+1:N,1))).*(1+exp(markov_chain(N_burnin+1:N,2)))));%posterior variance for alpha

pme_beta = mean(exp(markov_chain(N_burnin+1:N,1))./((1+exp(markov_chain(N_burnin+1:N,1)))...
    .*(1+exp(markov_chain(N_burnin+1:N,2)))));% posterior mean for  beta
psd_beta = std(exp(markov_chain(N_burnin+1:N,1))./((1+exp(markov_chain(N_burnin+1:N,1))).*...
    (1+exp(markov_chain(N_burnin+1:N,2)))));%posterior variance for beta

pme_w = mean(exp(markov_chain(N_burnin+1:N,3)));% posterior mean for w
psd_w = std(exp(markov_chain(N_burnin+1:N,3)));%posterior variance for w

%%  =================Forecasting volatility================================
y_911 = y(T); %stock return on 11 Sep 2020

%volatility on 11 Sep 2020
sigma_t_2 = zeros(T,1);
sigma_t_2(1) = var(y);
for i = 2:T
    sigma_t_2(i) = pme_w + pme_alpha*y(i-1)^2 + pme_beta*sigma_t_2(i-1);
end
sigma_911_2 = sigma_t_2(T);

posterior_mean_estimate = [pme_w ,pme_alpha,pme_beta];
posterior_standard_deviation = [psd_w, psd_alpha, psd_beta]
predictive_volatility_square = pme_w + pme_alpha*(y_911^2) + pme_beta*(sigma_911_2);%calculate predictive_mean 

%% ========================Maximum Likelihood estimation=================== 
Md10 = garch('Constant', 0.0001,'GARCH',0.5,'ARCH',0.2)
rng default

Md1 = garch(1,1)
EstMd1 = estimate(Md1,y)

mle_volatility_square = 4.65426e-06 + 0.0800751*(y_911^2) +0.895364*(sigma_911_2)%calculate predictive_mean 
