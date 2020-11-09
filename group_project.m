clear all
rng(2020) % Fix the random seed
price = readtable('Price')
l = height(price) %row length of the dataframe
%% 
a = log(price(1,2))
%% 

for i = 1:l
    r = log(price(i+1,2)) -log(price(i,2))
end