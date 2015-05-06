

%% EP-DGP and MCMC-DGP DEMO

load 'Datasets\Wahba.mat'

%The function provides NMSE, NMAE, and NLPD (measured on the test set) for the standard GP,
%EP-DGP and MCMC-DGP. If pl=1, the function plots the predicted mean with
%the error bars for the 3 methods (only available for unidimensional data
%sets).
pl = 1;
Results = DGP_ui(x_tr,y_tr,x_tst,y_tst,pl);


