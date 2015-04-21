function [nlpd] = predictNLPD(mu_f,var_f,mu_g,var_g,k,y)

%Negative Log-predictive Density calculation function
%INPUTS: 
% - mu_f: mean of latent function f
% - var_f: variance of latent function f
% - mu_g: mean of latent function g
% - var_g: variance of latent function g
% - k: division coefficient of EPDGP
% - y: target values
%
%OUTPUTS: 
%- nlpd: negative log-predictive density


mu_y = mu_g.*y;
var_y = k + var_f + var_g.*(y.^2);

gauss_y = (1./sqrt(2.*pi.*var_y)).*exp(-((mu_f - mu_y).^2)./(2.*var_y));

%Mean of a truncated Gaussian on g
t1 = (1./sqrt(2.*pi)).*exp(-(mu_g.^2)./(2.*var_g));
t2 = normcdf(0,-mu_g,sqrt(var_g));
mu_gt = mu_g + sqrt(var_g).*t1./t2;

%Predictive Density
pd = mu_gt.*gauss_y;

%NLPD
nlpd = -log(pd);