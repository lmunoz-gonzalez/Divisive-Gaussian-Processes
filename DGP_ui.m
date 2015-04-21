function [NMSE_EPDGP, NMAE_EPDGP, NLPD_EPDGP,NMSE_MCMC, NMAE_MCMC, NLPD_MCMC, NMSE_GP, NMAE_GP, NLPD_GP] = DGP_ui(x_tr,y_tr,x_tst,y_tst,pl);

% This function trains EP-DGP and MCMC-DGP for a training data given in
% x_tr and y_tst and evaluates the predictions for test data (given in
% x_tst and y_tst) in terms of NMSE, NMAE, and NLPD. To initialize the
% hyperparameters, a standard GP is trained first. The performance of the
% standard GP in terms of NMSE, NMAE, and NLPD is also provided.

%INPUTS: 
% - x_tr: input training data of dimension (# of samples x # of dimensions)
% - y_tr: input targets.
% - x_tst: input test data of dimension # (# of samples x # of dimensions)
% - pl: flag to show plot (only for unidimensional data sets).
%       pl=1 ->show plot
%       pl=0 ->don't show plot (default option)

%OUTPUTS: 
% - NMSE_EPDGP: Normalized mean squared error for EP-DGP method evaluated on the test set.
% - NMAE_EPDGP: Normalized mean absolute error for EP-DGP method evaluated on the test set.
% - NLPD_EPDGP: Negative Log Predictive Density for EP-DGP method evaluated on the test set. 
% - NMSE_MCMC: Normalized mean squared error for MCMC-DGP method evaluated on the test set.
% - NMAE_MCMC: Normalized mean absolute error for MCMC-DGP method evaluated on the test set.
% - NLPD_MCMC: Negative Log Predictive Density for MCMC-DGP method evaluated on the test set. 
% - NMSE_GP: Normalized mean squared error for the standard GP evaluated on the test set.
% - NMAE_GP: Normalized mean absolute error for the standard GP evaluated on the test set.
% - NLPD_GP: Negative Log Predictive Density for the standard GP evaluated on the test set. 

if nargin < 5
    pl = 0;
end

%Remove the training mean to the training targets
meanp = mean(y_tr);
y_tr = y_tr - meanp;

D = size(x_tr,2); %Dimension of the data

%% STANDARD GP
% Train a simple GP to initialize hyperparameters of DPG models

% ISO exponential covariance function
covfunc = {'covSum', {'covSEiso','covNoise'}};

% ARD exponential covariance function
% covfunc = {'covSum', {'covSEard','covNoise'}};

% Hyperparameter initialization
SignalPower = var(y_tr,1);
NoisePower = SignalPower/4;
% Lenghtscale initialization for ISO exponential covariance function
lengthscales=log(mean((max(x_tr)-min(x_tr))'/2));
% Lenghtscale initialization for ARD exponential covariance function
% lengthscales=log(((max(x_tr)-min(x_tr))'/2));

%Hyperparameters initial vector
loghyper_GP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower)];

%Train standard GP
fprintf('Training standard GP............\n');
loghyper_GP = minimize(loghyper_GP, 'gpr', 100, covfunc, x_tr, y_tr);

% Compute standard GP predictions
fprintf('Computing standard GP predictions\n');
[mu_tst_GP S2_tst_GP]= gpr(loghyper_GP, covfunc, x_tr, y_tr, x_tst);
mu_tst_GP = mu_tst_GP + meanp;

%Compute NMSE, NMAE, and NLPD on the test data
NMSE_GP = mean( (y_tst - mu_tst_GP).^2)/mean( (y_tst - meanp).^2);
NMAE_GP = mean(abs(y_tst - mu_tst_GP))/mean(abs(y_tst - meanp));
NLPD_GP = -0.5*mean( - ((mu_tst_GP - y_tst).^2)./(S2_tst_GP) - log(2*pi) - log(S2_tst_GP));

if ((D==1) && (pl==1))
    x_plot = linspace(min(x_tr),max(x_tr),150)';
    [mu_plot S2_plot]= gpr(loghyper_GP, covfunc, x_tr, y_tr, x_plot);
    mu_plot = mu_plot + meanp;
    figure
    plot(x_tr,y_tr + meanp,'+')
    hold on
    plot(x_plot,mu_plot)
    plot(x_plot,mu_plot + 2.*sqrt(S2_plot),'--')
    plot(x_plot,mu_plot - 2.*sqrt(S2_plot),'--')
end


%% EP-DGP

%Covariance functions (with ISO exponential kernel)
covfunc1 = {'covSum', {'covSEiso','covNoise'}};   %Covariance function for latent function f
covfunc2 = {'covSEiso2'}; %Covariance function for latent function g

%Covariance functions (with ARD exponential kernel)
% covfunc1 = {'covSum', {'covSEard','covNoise'}};   %Covariance function for latent function f
% covfunc2 = {'covSEard2'}; %Covariance function for latent function g

%Hyperparameter initialization
NoisePower = exp(2.*loghyper_GP(end));
SignalPower = exp(2.*loghyper_GP(end-1));
lengthscales = loghyper_GP(1:end-2);
mu0 = 2/sqrt(NoisePower);
k_noise = 4;
Gpower = (mu0^2)/10;
loghyperNoise =  [lengthscales; 0.5*log(Gpower); log(mu0)];
Fpower = SignalPower*(mu0^2)/k_noise;
loghyperSignal = [lengthscales; 0.5*log(Fpower); 0.5*log(Fpower/4)];
size_loghyperSignal = length(loghyperSignal);

%Vector of initial hyperparameters
loghyper = [loghyperSignal; loghyperNoise];

%Train EP-DGP
fprintf('\nTraining EP-DGP............\n');
[newloghyper] = minimize(loghyper,'EPDGP',80,covfunc1,covfunc2,k_noise,x_tr,y_tr);

%Predictions of EP-DGP on the test data
fprintf('Computing EP-DGP predictions\n');
[mu_f s2_f mu_g s2_g nlml] = EPDGP(newloghyper, covfunc1, covfunc2, k_noise, x_tr, y_tr, x_tst);
%Prediction of the median
mu_tst = predictMedian(mu_f,s2_f,mu_g,s2_g,k_noise); 
mu_tst = mu_tst +  meanp;

%Compute NMSE, NMAE, and NLPD on the test data
NMSE_EPDGP = mean((y_tst - mu_tst).^2)/mean((y_tst - meanp).^2);
NMAE_EPDGP = mean(abs(y_tst - mu_tst))/mean(abs(y_tst - meanp));
[nlpd] = predictNLPD(mu_f,s2_f,mu_g,s2_g,k_noise,y_tst-meanp);
NLPD_EPDGP = mean(nlpd); 

%Plot predictions if data sets is unidimensional
if ((D==1) && (pl==1))
    [mu_f s2_f mu_g s2_g nlml] = EPDGP(newloghyper, covfunc1, covfunc2, k_noise, x_tr, y_tr, x_plot);
    mu_plot = predictMedian(mu_f,s2_f,mu_g,s2_g,k_noise) + meanp;
    %Predict quantiles 0.1 and 0.9
    upperQ_plot = predictQuantile(mu_f,s2_f,mu_g,s2_g,k_noise,0.95) + meanp;
    lowerQ_plot = predictQuantile(mu_f,s2_f,mu_g,s2_g,k_noise,0.05) +meanp;
    plot(x_plot,mu_plot,'r')
    plot(x_plot,upperQ_plot,'--r')
    plot(x_plot,lowerQ_plot,'--r')
end


%% MCMC-DGP

%Covariance functions (with ISO exponential kernel)
covfunc1 = {'covSum', {'covSEiso','covNoise'}};   %Covariance function for latent function f
covfunc2 = {'covSEiso2'}; %Covariance function for latent function g

%Covariance functions (with ARD exponential kernel)
% covfunc1 = {'covSum', {'covSEard','covNoise'}};   %Covariance function for latent function f
% covfunc2 = {'covSEard2'}; %Covariance function for latent function g

loghyperSignal = newloghyper(1:size_loghyperSignal);
loghyperNoise = newloghyper(size_loghyperSignal+1: end-1);

Kf = feval(covfunc1{:},loghyperSignal, x_tr);
Kg = feval(covfunc2{:},loghyperNoise, x_tr);
%Kg = feval('covSEiso',loghyperNoise, x_tr);
mu0 = exp(newloghyper(end));
k_noise = 4;

n = length(x_tr);
Kf = Kf + 1e-7.*eye(n);
Kg = Kg + 1e-7.*eye(n);

% TRAINING PHASE
fprintf('\nTraining MCMC-DGP............\n');
mcmcOps.BurnIn = 5000;  % burn n iterations 
mcmcOps.T = 25000;    % main sampling iterations 
mcmcOps.StoreEvery = 5;  % use for Monte Carlo estimates T/StoreEvery "thinned" samples
% run the sampler to collect the samples

samples =  dgpMCMC(x_tr, y_tr, Kf, Kg, mu0,k_noise, mcmcOps);

% TESTING PHASE
fprintf('Computing MCMC-DGP predictions\n');
[Kfss, Kfstar] = feval(covfunc1{:}, loghyperSignal,x_tr, x_tst); % test covariance f
[Kgss, Kgstar] = feval(covfunc2{:}, loghyperNoise,x_tr, x_tst); % test covariance g

% Compute predictive mean and variance 
[mu_f, var_f,mu_g,var_g,samples2] = dgpMCMC(x_tr, y_tr, Kf, Kg, mu0,k_noise,[], samples, Kfss, Kgss, Kfstar, Kgstar);
mu_g = mu_g + mu0;
% Evaluate results
mu_tst = predictMedian(mu_f,var_f,mu_g,var_g,k_noise); 
mu_tst = mu_tst +  meanp;

%Calculate NMSE, NMAE, and NLPD
NMSE_MCMC = mean((y_tst - mu_tst).^2)/mean((y_tst - meanp).^2);
NMAE_MCMC = mean(abs(y_tst - mu_tst))/mean(abs(y_tst - meanp));
[nlpd] = predictNLPD(mu_f,var_f,mu_g,var_g,k_noise,y_tst-meanp);
NLPD_MCMC = mean(nlpd); 

if ((D==1) && (pl==1))
    [Kfss, Kfstar] = feval(covfunc1{:}, loghyperSignal,x_tr, x_plot); % test covariance f   
    [Kgss, Kgstar] = feval(covfunc2{:}, loghyperNoise,x_tr, x_plot); % test covariance g  
    % Compute predictive mean and variance 
    [mu_f_plot, s2_f_plot,mu_g_plot,s2_g_plot,samples2] = dgpMCMC(x_tr, y_tr, Kf, Kg, mu0,k_noise,[], samples, Kfss, Kgss, Kfstar, Kgstar);
    mu_g_plot = mu_g_plot + mu0;
    mu_plot = predictMedian(mu_f_plot,s2_f_plot,mu_g_plot,s2_g_plot,k_noise) + meanp;
    %Predict quantiles 0.05 and 0.95
    upperQ_plot = predictQuantile(mu_f_plot,s2_f_plot,mu_g_plot,s2_g_plot,k_noise,0.95) + meanp;
    lowerQ_plot = predictQuantile(mu_f_plot,s2_f_plot,mu_g_plot,s2_g_plot,k_noise,0.05) + meanp;
    plot(x_plot,mu_plot,'k')
    plot(x_plot,upperQ_plot,'--k')
    plot(x_plot,lowerQ_plot,'--k')
    title('Standard GP prediction (BLUE), EP-DGP predictions (RED), MCMC-DGP predictions (BLACK)');
end

