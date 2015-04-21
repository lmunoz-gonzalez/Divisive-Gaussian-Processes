function [out1, out2, out3, out4, out5] = EPDGP(logtheta, covfunc1,covfunc2, k_noise, x, y, xstar);

% EXPECTATION PROPAGATION DIVISIVE GAUSSIAN PROCESSES 

% Two modes are possible: training or testing: if no
% test cases are supplied, then the approximate negative log marginal
% likelihood and its partial derivatives w.r.t. the hyperparameters is computed;
% this mode is used to fit the hyperparameters. If test cases are given, then
% the test set predictive probabilities are returned. The program is flexible
% in allowing a multitude of covariance functions.
%
% usage: [nlml dnlml] = EPDGP(logtheta, covfunc1, covfunc2, k_noise, x, y);
%    or: [mu_f s2_f mu_g s2_g nlml] = EPDGP(logtheta, covfunc1, covfunc2, k_noise, x, y, xstar);
%
% where:
%
%   logtheta is a (column) vector of hyperparameters
%   covfunc1 is the name of the signal covariance function (see below)
%   covfunc2 is the name of the log-noise covariance function(see below)
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the sum of the negative log marginal 
%            likelihoods nlml_f, nlml_g   
%   dnlml    is a (column) vector of partial derivatives of nlml
%            wrt each log hyperparameter (on f and g)
%   mu_f     is a (column) vector (of length nn) with the mean of the
%            signal latent function f
%   s2_f     is a (column) vector (of length nn) withe the variance of
%            the signal latent function f
%   mu_g     is a (column) vector (of length nn) with the mean of the
%            log-noise latent function g
%   s2_g     is a (column) vector (of length nn) withe the variance of
%            the log-noise latent function g
%
% The length of the vector of log hyperparameters depends on the covariance
% function, as specified by the "covfunc1" and "Covfunc2" inputs, specifying the
% name of a covariance function. A number of different covariance function are
% implemented, and it is not difficult to add new ones. See "help covFunctions"
% for the details
%
% The function can conveniently be used with the "minimize" function to train
% a Gaussian process:
% e.g.:
% [logtheta, fX, i] = minimize(logtheta, 'EPDGP', covfunc1,
% covfunc2,k_noise, x, y);
%
%
% Author: Luis Muñoz-González (Based on Carl Edward Rasmussen's GPML code).
% June 2013


%% Hyperparameters checkings
if ischar(covfunc1), covfunc1 = cellstr(covfunc1); end % convert to cell if needed
if ischar(covfunc2), covfunc2 = cellstr(covfunc2); end % convert to cell if needed

%Check the number of hyperparameters for latent function f
[n, D] = size(x);

n_logtheta_f = eval(feval(covfunc1{:})); %Number of hyperparameters on f
n_logtheta_g = eval(feval(covfunc2{:})); %Number of hyperparameters on g

if ((n_logtheta_f + n_logtheta_g  + 1) ~= length(logtheta))
    error('Error: Number of parameters do not agree with covariance function g')
end

logtheta_f = logtheta(1:n_logtheta_f);
logtheta_g = logtheta(n_logtheta_f+1:end-1);
mu0 = exp(logtheta(end));

%% Persistent variables
persistent best_ttau_f best_ttau_g best_tnu_f best_tnu_g...
    best_lml_f best_lml_g best_Z_hat;   % keep tilde parameters between calls

%% Evaluation of covariance Matrices given the sets of hyperparameters
K_f = feval(covfunc1{:}, logtheta_f, x);                % the covariance matrix on f
K_g = feval(covfunc2{:}, logtheta_g, x);                % the covariance matrix on g                               
L_kg = chol(K_g  + eye(n).*1e-6);                      % Cholesky factorization of covariance matrix K_g
mu0vector = mu0.*ones(n,1);                             % vector with the mean of the prior on g

%% NAMING VARIABLES
% A note on naming: variables are given short but descriptive names in 
% accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
% and s2 are mean and variance, nu and tau are natural parameters. A leading t
% means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
% for a vector of cavity parameters. Subscripts _f and _g refers to signal
% latent function f and log-noise latent function g respectively.


%% INITIALIZATION ON F
if any(size(best_ttau_f) ~= [n 1])     % find starting point for tilde parameters
  ttau_f = zeros(n,1);            % initialize to zero if we have no better guess
  tnu_f = zeros(n,1);
  mu_f = zeros(n,1);
  Sigma_f = K_f;          % initialize Sigma, the variance of the posterior approximation
  lml_f = -inf;
  best_lml_f = -inf;
else
  ttau_f = best_ttau_f;                   % try the tilde values from previous call
  tnu_f = best_tnu_f;
  [Sigma_f, mu_f, lml_f] = epComputeParams_f(K_f, y, ttau_f, tnu_f); 
end

%% INITIALIZATION ON G
if any(size(best_ttau_g) ~= [n 1])     % find starting point for tilde parameters
  ttau_g = zeros(n,1);            % initialize to zero if we have no better guess
  tnu_g = zeros(n,1);
  Sigma_g = K_g;                    % initialize Sigma and mu, the parameters of ..
  mu_g = mu0vector;                   % .. the Gaussian posterior approximation
  lml_g = -inf;
  best_lml_g = -inf;
  Z_hat = ones(n,1);           
else
  ttau_g = best_ttau_g;                   % try the tilde values from previous call
  tnu_g = best_tnu_g;
  Z_hat = best_Z_hat;
  [Sigma_g, mu_g, lml_g] = epComputeParams_g(K_g,L_kg, y, ttau_g, tnu_g,Z_hat,mu0vector); 
end

%% EP convergence parameters
tol = 1e-4;                 % tolerance level to stop EP iterations
max_sweep = 150;             % maximum number of EP sweeps

%% EP LOOP
sweep = 0;                        % make sure while loop starts
dif = 1e6;

while dif > tol & sweep < max_sweep   % converged or maximum sweeps?
  %Store and update EP convergence control parameters
  lml_old_f = lml_f; 
  lml_old_g = lml_g; 
  sweep = sweep + 1;
  
  for i = randperm(n)          % iterate EP updates over examples permuted randomly   
    %----------------------------------
    % CAVITY PARAMETERS
    %----------------------------------
    %First find the cavity parameters tau_ni_f and nu_ni_f
    tau_ni_f = 1/Sigma_f(i,i) - ttau_f(i);
    nu_ni_f = mu_f(i)/Sigma_f(i,i) - tnu_f(i);   

    %Find the cavity distribution parameters tau_ni_g and nu_ni_g
    tau_ni_g = 1/Sigma_g(i,i) - ttau_g(i);      
    nu_ni_g = mu_g(i)/Sigma_g(i,i) - tnu_g(i);

    %------------------------------
    % MOMENTS CALCULATION
    %------------------------------
    var_ni_g = 1/tau_ni_g; %Covariance matrix of the cavity function on g
    mu_ni_g = var_ni_g * nu_ni_g; %Mean vector of the cavity function on g
    var_ni_f = 1/tau_ni_f; %Covariance matrix of the cavity function on f
    mu_ni_f = var_ni_f * nu_ni_f; %Mean vector of the cavity function on f
    [Z_est, mu_est, var_est] = momentsEstimation(y(i), mu_ni_f, var_ni_f, mu_ni_g, var_ni_g, k_noise);
    m_est_f = mu_est(1); 
    m_est_g = mu_est(2);
    v_est_f = var_est(1,1);
    v_est_g = var_est(2,2);  
    
    %---------------------------------------
    % SITE FUNCTIONS UPDATE
    %---------------------------------------
    %Site functions on f
    ttau_old_f = ttau_f(i);  
    ttau_f(i) = 1/v_est_f - tau_ni_f; 
    tnu_f(i) = m_est_f/v_est_f - nu_ni_f;
    
    %Site functions on g
    ttau_old_g = ttau_g(i);
    ttau_g(i) = 1/v_est_g - tau_ni_g;
    tnu_g(i) = m_est_g/v_est_g - nu_ni_g;
    Z_hat(i) = Z_est;    
    
    %---------------------------------------------
    % RANK-1 UPDATE OF SIGMA_F AND SIGMA_G
    % This part is not strictly necessary. It consumes about 70% of
    % the loop time. However, this rank 1 updates favors algorithm
    % stability.
    %---------------------------------------------
    %Sigma_f rank-1 update
    ds2_f = ttau_f(i) - ttau_old_f;                  % finally rank-1 update Sigma ..
    Sigma_f = Sigma_f - ds2_f/(1+ds2_f*Sigma_f(i,i))*Sigma_f(:,i)*Sigma_f(i,:);
    mu_f = Sigma_f*tnu_f;     
    
    %Sigma_g rank-1 update    
    ds2_g = ttau_g(i) - ttau_old_g;                  % finally rank-1 update Sigma ..
    Sigma_g = Sigma_g - ds2_g/(1+ds2_g*Sigma_g(i,i))*Sigma_g(:,i)*Sigma_g(i,:);
    %mu_g = Sigma_g*(tnu_g + invK_g*mu0vector);    
    mu_g = Sigma_g*(tnu_g + L_kg\(L_kg'\mu0vector));
    %----------------------------------------------
    
  end
  
  %-------------------------------------------------
  %RECALCULTION OF POSTERIOR PARAMETERS
  %-------------------------------------------------
  
  %Recompute Sigma_f and mu_f since repeated rank-one updates eventually
  %destroys numerical precision
  [Sigma_f, mu_f, lml_f, L_f] = epComputeParams_f(K_f, y, ttau_f, tnu_f);      
  
  %Recompute Sigma_g and mu_g since repeated rank-one updates eventually
  %destroys numerical precision
  [Sigma_g, mu_g, lml_g, L_g] = epComputeParams_g(K_g,L_kg, y, ttau_g, tnu_g,Z_hat,mu0vector);  

  %Check convergence
  lml_old = lml_old_f + lml_old_g;     
  lml_new = lml_f + lml_g;
  dif = abs(lml_new - lml_old); 
end


if sweep == max_sweep   
  disp('Warning: maximum number of sweeps reached in function EPDGP')
end

%% STORE VALUES IF THE ALGORITHM IMPROVES LOG-EVIDENCE
if ((lml_f + lml_g) > (best_lml_f + best_lml_g))
  best_ttau_f = ttau_f; best_tnu_f = tnu_f; best_lml_f = lml_f; 
  best_ttau_g = ttau_g; best_tnu_g = tnu_g; best_lml_g = lml_g;
  best_Z_hat = Z_hat; 
end


%% COMPUTATION OF LOG-EVIDENCE DERIVATIVES WITH RESPECT TO THE
%% HYPERPARAMETERS
if nargin == 6                   % return the negative log marginal likelihood?

  out1 = -lml_f - lml_g;

  if nargout > 1                                      % do we want derivatives?
    out2 = 0*logtheta;                         % allocate space for derivatives   
    
    %Calculate derivatives on f
    b = tnu_f-sqrt(ttau_f).*solve_chol(L_f,sqrt(ttau_f).*(K_f*tnu_f));
    F = b*b'-repmat(sqrt(ttau_f),1,n).*solve_chol(L_f,diag(sqrt(ttau_f)));    
    for j=1:length(logtheta_f)
      C_f = feval(covfunc1{:}, logtheta_f, x, j);
      out2(j) = -sum(sum(F.*C_f))/2;
    end
    
    %Calculate derivatives on g
    nu_g = ttau_g.*mu0vector - tnu_g;
    b = nu_g-sqrt(ttau_g).*solve_chol(L_g,sqrt(ttau_g).*(K_g*nu_g));
    F = b*b'-repmat(sqrt(ttau_g),1,n).*solve_chol(L_g,diag(sqrt(ttau_g)));        
    for jj=1:length(logtheta_g)
        j = j+1;
        C_g = feval(covfunc2{:}, logtheta_g, x, jj);
        out2(j) = -sum(sum(F.*C_g))/2;
    end 
    
    %Derivative for mu_0
    j = j+1;
    out2(j) = sum(b).*mu0;
    
    if nargout == 6, out3 = ttau_f; out4 = tnu_f; out5 = ttau_g; out6 = tnu_g; end     % return the tilde params
  end
    
else                               % otherwise compute predictive probabilities
  
%% PREDICTIONS

  %Predictions on f
  %-------------------------------------------------
  [a b] = feval(covfunc1{:}, logtheta_f, x, xstar);
  
  %Test mean on f
  mus_f = b'*(tnu_f-sqrt(ttau_f).*solve_chol(L_f,sqrt(ttau_f).*(K_f*tnu_f)));
  
  %Test variance on f
  v_f = L_f'\(repmat(sqrt(ttau_f),1,size(xstar,1)).*b);
  s2s_f = a - sum(v_f.*v_f,1)'; 
  
  %Predictions on g
  %-------------------------------------------------
  [a b] = feval(covfunc2{:}, logtheta_g, x, xstar);
  
  tnu_g2 = tnu_g - ttau_g.*mu0vector;
  mus_g = b'*(tnu_g2-sqrt(ttau_g).*solve_chol(L_g,sqrt(ttau_g).*(K_g*tnu_g2)));
  mus_g = mus_g + mu0; 
  
  %Test variance on g
  v_g = L_g'\(repmat(sqrt(ttau_g),1,size(xstar,1)).*b);
  s2s_g = a - sum(v_g.*v_g,1)'; 
  
  %Outputs
  %-------------------------------------------------
  out1 = mus_f;  %Test means on f
  out2 = s2s_f;  %Test variance on f
  out3 = mus_g; %Test means on g
  out4 = s2s_g;  %Test variance on g
  out5 = -lml_f - lml_g; %Negative log marginal likelihood

end


end

%% epComputeParams on f
% function to compute the parameters of the Gaussian approximation, Sigma and
% mu, and the log marginal likelihood, lml, from the current site parameters,
% ttau and tnu. The function also may return L (useful for predictions).

function [Sigma, mu, lml, L] = epComputeParams_f(K, y, ttau, tnu); 
    n = length(y);                                       % number of training cases
    ssi = sqrt(ttau);                                        % compute Sigma and mu
    L = chol(eye(n)+ssi*ssi'.*K);
    V = L'\(repmat(ssi,1,n).*K);
    Sigma = K - V'*V;
    mu = Sigma*tnu;   
    tau_n = 1./diag(Sigma)-ttau;              % compute the log marginal likelihood
    nu_n = mu./diag(Sigma)-tnu;                      % vectors of cavity parameters  
    lml   = - sum(log(diag(L)))+tnu'*Sigma*tnu/2 + nu_n'*((ttau./tau_n.*nu_n-2*tnu)./(ttau+tau_n))/2 ...
           -sum(tnu.^2./(tau_n+ttau))/2 + sum(log(1+ttau./tau_n))/2;
  
end 



%% epComputeParams on g
% function to compute the parameters of the Gaussian approximation, Sigma and
% mu, and the log marginal likelihood, lml, from the current site parameters,
% ttau and tnu. The function also may return L (useful for predictions).

function [Sigma, mu, lml, L] = epComputeParams_g(K,L_k, y, ttau, tnu,Z_hat,mu0vector); 
    n = length(y);                                       % number of training cases
    ssi = sqrt(ttau);                                        % compute Sigma and mu
    L = chol(eye(n)+ssi*ssi'.*K);
    V = L'\(repmat(ssi,1,n).*K);
    Sigma = K - V'*V;
    mu = Sigma*(tnu + L_k\(L_k'\mu0vector));

    tau_n = 1./diag(Sigma)-ttau;              % compute the log marginal likelihood
    nu_n = mu./diag(Sigma)-tnu;                      % vectors of cavity parameters   
    lml = -sum(log(diag(L)))+sum(log(1+ttau./tau_n))/2+sum(log(Z_hat)) ...
          +tnu'*Sigma*tnu/2+nu_n'*((ttau./tau_n.*nu_n-2*tnu)./(ttau+tau_n))/2 ...
          -sum(tnu.^2./(tau_n+ttau))/2 -0.5.*mu0vector'*(L_k\(L_k'\Sigma))*(ttau.*mu0vector - 2.*tnu);
end
 
  