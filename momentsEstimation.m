function [Z_est, mu_est, var_est] = momentsEstimation(yn, mu_ni_f, var_ni_f, mu_ni_g, var_ni_g, k);

% Function to calculate the moments for EP Divisive GP. 
% INPUTS: 
% - yn: n-th target sample
% - mu_ni_f: mean of the cavity distribution on fn
% - var_ni_f: variance of the cavity distribution on fn
% - mu_ni_g: mean of the cavity distribution on gn
% - var_ni_g: variance of the cavity distribution on gn
% - k: noise constant
% 
% OUTPUTS: 
% - Z_est: zeroth order moment estimation
% - mu_est: first order moment estimation
% - var_est: estimated covariance matrix


%% MOMENTS OF THE TRUNCATED GAUSSIAN
var_g = 1/(1/var_ni_g + yn^2/(k + var_ni_f));
mu_g = var_g*(mu_ni_g/var_ni_g + (mu_ni_f*yn)/(k + var_ni_f));

sigma = sqrt(var_g);
%Cnorm = normcdf(0,-mu_g,sigma);
Cnorm = 0.5 * erfc((-mu_g/sigma) ./ sqrt(2));

h = mu_g/sigma;
lambda = (1/sqrt(2*pi)).*exp(-((mu_g)^2)/(2*sigma^2));
lambda = lambda/Cnorm;

I1 = -lambda;
I2 = -h*lambda + 1;
I3 = -lambda*(h^2 + 2);

%Mean of the truncated Gaussian
mu_tr = mu_g - sigma*I1;
%2nd order moment of the truncated Gaussian
m2 = mu_g^2 - 2*mu_g*sigma*I1 + (sigma^2)*I2;
%Variance of the truncated Gaussian
var_tr = m2 - mu_tr^2;
%3rd order moment of the truncated Gaussian
m3_tr = mu_g^3 - 3*(mu_g^2)*sigma*I1 + 3*mu_g*(sigma^2)*I2 - (sigma^3)*I3;


%% ZEROTH ORDER MOMENT
var_gy = k + var_ni_f + var_ni_g*(yn^2);
Z_gy = (1/sqrt(2*pi*var_gy))*exp(-((mu_ni_g*yn - mu_ni_f)^2)/(2*var_gy));

%Zeroth order moment
Z_est = Z_gy*mu_tr*Cnorm;


%% FIRST ORDER MOMENT ON gn
mu_gn_est = Z_gy*(var_tr + mu_tr^2);
mu_gn_est = mu_gn_est*Cnorm/Z_est;


%% FIRST ORDER MOMENT ON fn
var_f = 1/(1/k + 1/var_ni_f);
term1 = Z_est*mu_ni_f*var_f/var_ni_f; 
term2 = var_f*yn*Z_est*mu_gn_est/k;
mu_fn_est = (term1 + term2)/Z_est; 


%% SECOND ORDER MOMENT ON gn
v2_gn_est = Z_gy*m3_tr*Cnorm/Z_est;

%% SECOND ORDER MOMENT ON fn
term1 = var_f;
term2 = v2_gn_est*(var_f^2)*(yn^2)/(k^2);
term3 = ((var_f*mu_ni_f)^2)/(var_ni_f^2);
term4 = mu_gn_est*2*(var_f^2)*yn*mu_ni_f/(k*var_ni_f);
v2_fn_est = (term1 + term2 + term3 + term4);


%% GROUPING TERMS
mu_est = [mu_fn_est; mu_gn_est];
var_fn_est = v2_fn_est - mu_fn_est^2;
var_gn_est = v2_gn_est - mu_gn_est^2;
var_est = [var_fn_est 0; 0 var_gn_est];


