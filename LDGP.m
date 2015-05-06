function [out1, out2, out3, out4, out5] = LDGP(logtheta, covfunc1,covfunc2, k_noise, x, y, xstar);

% LaplaceDGP - Laplace's approximation for Divisive GP model. Two modes are 
% possible: training or testing: if no test
% cases are supplied, then the approximate negative log marginal likelihood
% and its partial derivatives w.r.t. the hyperparameters is computed; this mode is
% used to fit the hyperparameters. If test cases are given, then the mean and 
% variance of latent function f and g are provided. The program is flexible
% in allowing several covariance functions. 
%
% usage: [nlml dnlml] = LaplaceDGP(logtheta, covfunc1,covfunc2, k_noise, x, y);
%    or: [mu_f s2_f mu_g s2_g nlml] = LaplaceDGP(logtheta, covfunc1, covfunc2, k_noise, x, y, xstar);
%
% where:
%
%   logtheta is a (column) vector of hyperparameters
%   covfunc1 is the name of the covariance function for f 
%   covfunc2 is the name of the covariance function for g
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml   is the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of nlml
%            wrt each log hyperparameter (on f and g)
%   mu_f     is a (column) vector (of length nn) with the mean of latent 
%            function f
%   s2_f     is a (column) vector (of length nn) withe the variance of
%            latent function f
%   mu_g     is a (column) vector (of length nn) with the mean of latent 
%            function g
%   s2_g     is a (column) vector (of length nn) withe the variance of
%            latent function g
%
% The length of the vector of log hyperparameters depends on the covariance
% functions, as specified by the "cov" inputs to the function, specifying the 
% name of the 2 covariance functions. A number of different covariance function are
% implemented, and it is not difficult to add new ones. See "help covFunctions"
% for the details.
%
% The function can conveniently be used with the "minimize" function to train
% a Gaussian process:
%
% [logtheta, fX, i] = minimize(logtheta, 'LaplaceDGP', length, 'covSEiso',
%                              'covSEiso2', k_noise,x,y);
%
% Note, that the function has a set of persistent variables where the best "a"
% vector so far and the value of the corresponding approximate negative log 
% marginal likelihood is recorded. The Newton iteration is started from this
% guess (if it isn't worse than zero), which should mostly be quite reasonable
% guesses when the function is called repeatedly (from eg "minimize"), when
% finding good values for the hyperparameters.
%
% Author: Luis Muñoz-González. Based on Carl Rasmussen's GPML implementation.


%% Hyperparameters checkings

%Check the number of hyperparameters for latent functions f and g
[n, D] = size(x);

n_logtheta_f = eval(feval(covfunc1{:})); %Number of hyperparameters on f
n_logtheta_g = eval(feval(covfunc2{:})); %Number of hyperparameters on g

if ((n_logtheta_f + n_logtheta_g  + 1) ~= length(logtheta))
    error('Error: Number of parameters do not agree with the covariance functions.')
end

logtheta_f = logtheta(1:n_logtheta_f);
logtheta_g = logtheta(n_logtheta_f+1:end-1);
mu0 = exp(logtheta(end));


%% Persistent values and tolerance settings for Newton algorithm
persistent best_a best_value;   % keep a copy of the best "a" and its obj value
tol = 1e-10;                  % tolerance for when to stop the Newton iterations
[n D] = size(x);             % size of training data

%% Evaluation of covariance Matrices given the sets of hyperparameters
K_f = feval(covfunc1{:}, logtheta_f, x);                % the covariance matrix on f
K_g = feval(covfunc2{:}, logtheta_g, x);                % the covariance matrix on g                               
mu0vector = [zeros(n,1); mu0.*ones(n,1)];               % vector with the mean of the prior on f and g

%Joint covariance matrix on f and g (it is a block diagonal matrix as we
%are assumen prior independence between f and g
K = [K_f zeros(n); zeros(n) K_g]; 


%% Initialization
if any(size(best_a) ~= [2*n 1])      % find a good starting point for "a" and "phi"
  phi = zeros(2*n,1); 
  a = phi; 
  [lp dlp W] = likDGP(phi+mu0vector,y,k_noise);       %start at zero
  Psi_new = -1e10;
  best_value = inf;
else
  a = best_a; 
  phi = K*a; 
  [lp dlp W] = likDGP(phi+mu0vector,y,k_noise); % try the best "a" so far
  Psi_new = -a'*phi/2 + lp;         
  if Psi_new < -n*log(2)                                  % if zero is better..
    phi = zeros(2*n,1);
    a = phi; 
    [lp dlp W] = likDGP(phi+mu0vector,y,k_noise);  % ..then switch back
    Psi_new = -a'*phi/2 + lp;
  end
end
Psi_old = -inf;                                   % make sure while loop starts

%% Newton's method
iter = 0;
while Psi_new - Psi_old > tol                       % begin Newton's iterations
  iter = iter + 1;
  Psi_old = Psi_new; 
  a_old = a;
  Lw = chol_tridiag(W);
  L = chol(eye(2*n) + Lw*K*Lw');
  Wmatrix = [diag(W(1:n)) diag(W(2*n+1:end)); diag(W(2*n+1:end)) diag(W(n+1:2*n))];  
  b = Wmatrix*phi+dlp;
  a = b - Lw'*solve_chol(L,Lw*(K*b)); 
  
  phi = K*a;
  [lp dlp W d3lp] = likDGP(phi+mu0vector,y,k_noise);
  Psi_new = -a'*phi/2 + lp;
  
  i = 0;  
  while i < 20 & Psi_new < Psi_old               % if objective didn't increase
    a = (a_old+a)/2;                                 % reduce step size by half
    phi = K*a;
    [lp dlp W d3lp] = likDGP(phi+mu0vector,y,k_noise);
    Psi_new = -a'*phi/2 + lp;
    i = i + 1;
  end
end                                                   % end Newton's iterations

Lw = chol_tridiag(W);
L = chol(eye(2*n) + Lw*K*Lw');

nlmarglik = a'*phi/2 - lp + sum(log(diag(L)));      % approx neg log marginal lik
if nlmarglik < best_value                                   % if best so far...
  best_a = a; best_value = nlmarglik;          % ...then remember for next call
end


%% Compute negative log-marginal likelihood and derivatives w.r.t.
%% hyperparameters

if nargin == 6                   % return the negative log marginal likelihood?

  out1 = nlmarglik;    
  if nargout == 2                                     % do we want derivatives?        
    out2 = 0*logtheta;
    Z = Lw'*solve_chol(L,Lw);
    s2 = -0.5*diag(K - K*Z*K).*d3lp; 
        
    %Derivatives of the hyperparameters on f
    for j=1:length(logtheta_f)
      C_f = feval(covfunc1{:}, logtheta_f, x, j); 
      s1 = a(1:n)'*C_f*a(1:n)/2-sum(sum(Z(1:n,1:n).*C_f))/2;
      b = [C_f zeros(n);zeros(n) zeros(n)]*dlp;
      s3 = b-K*(Z*b);
      out2(j) = -s1 -s2'*s3;      
    end
    %Derivatives of the hyperparameters on g
    for j=1:length(logtheta_g)
      C_g = feval(covfunc2{:}, logtheta_g, x, j); 
      s1 = a(n+1:end)'*C_g*a(n+1:end)/2-sum(sum(Z(n+1:end,n+1:end).*C_g))/2;
      b = C_g*dlp(n+1:end);
      s3 = b-K(n+1:end,n+1:end)*(Z(n+1:end,n+1:end)*b);
      out2(j+length(logtheta_f)) = -s1-s2(n+1:end)'*s3;     
    end
     
    %Derivatives of hyperparameter mu0
    [d_explicit1,t_implicit] = dmu0(phi+mu0vector, y, k_noise);
    d_explicit2 = sum(s2);
    b =  -K*t_implicit;
    d_implicit =  -s2'*( b-K*(Z*b) );
    out2(end) = (-d_explicit1 - d_explicit2 - d_implicit).*mu0; 
  end

%% Compute predictions

else

  [a_f b_f] = feval(covfunc1{:}, logtheta_f, x, xstar);
  [a_g b_g] = feval(covfunc2{:}, logtheta_g, x, xstar);
  
  %Predictive model for f and g together
  a = [a_f; a_g];
  b = [b_f zeros(size(b_f));zeros(size(b_g)) b_g];
  v = L'\(Lw*b);
  mu = b'*dlp;
  s2 = a - sum(v.*v,1)';
  nstar = size(xstar,1);
  out1 = mu(1:nstar);
  out2 = s2(1:nstar);
  out3 = mu(nstar+1:end) + mu0;
  out4 = s2(nstar+1:end);
   
  %Return negative log-evidence
  out5 = nlmarglik;
end

end

%% LIKELIHOOD FUNCTION

% DGP likelihood function. The log likelihood (scalar), its derivatives
% (vector), negative 2nd derivatives (vector) and 3rd derivatives (vector) w.r.t. to f
% and g are provided. The 2nd and 3rd derivatives are represented as
% vectors, since the "cross-terms" are zero, as the likelihood factorizes 
% over cases and likelihood independence between f and g is assumed. 

function [lp, dlp, d2lp, d3lp] = likDGP(phi, y, k_noise); 
  n = length(phi)/2; %Number of samples
  f = phi(1:n);
  g = phi(n+1:end);
  %Take the positive part of g
  g(g<0) = 1e-5;
  %Log-likelihood
  lp = sum( log(abs(g)) - 0.5*log(2*pi*k_noise) - 0.5.*((y.*g - f).^2)./k_noise); 
  %Evaluate on f
  dlp_f = (y.*g - f)./k_noise;
  d2lp_f = ones(size(f)).*(-1/k_noise);
  d3lp_f = zeros(size(f));
  %Evaluate on g
  dlp_g = 1./g - (y.*(y.*g - f))./k_noise;
  d2lp_g = -1./g.^2 - (y.^2./k_noise); 
  d3lp_g = 2./g.^3;
  
  %Partial derivative on f and g (i.e.  \partial lp / (\partial f \partial g))
  dfg = y./k_noise;
  
  %Derivatives vectors
  dlp = [dlp_f; dlp_g];
  d2lp = [-d2lp_f; -d2lp_g; -dfg];
  d3lp = [d3lp_f; d3lp_g];   

end
  

function [d_explicit1,term_implicit] = dmu0(phi, y, k_noise); 
    n = length(phi)/2; %Number of samples
    f = phi(1:n);
    g = phi(n+1:end);
    g(g<0) = 1e-5;
    d_explicit_v = 1./g + y.*(f - y.*g)./k_noise;
    d_explicit1 = sum(d_explicit_v);
    
    t_imp_f = y./k_noise;
    t_imp_g = -y.^2./k_noise - 1./g.^2;
    term_implicit = [t_imp_f;t_imp_g];
    
end
