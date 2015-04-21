function [out1, out2, out3, out4, samples] =  dgpMCMC(X, y, Kf, Kg, mu0, k, ops, samples, Kfss, Kgss, Kfstar, Kgstar, ystar)
%Description:  MCMC for the divisive GP model, where sampling is done using
%elliptical slice sampling. Two modes are provided. 
% 
%Training mode: It samples from the model p(y,g,f) = N(y|f/pos(g), k/pos(g)^2) N(g|mu0, Kg) N(f|0,Kf)       
%or equivalently p(y,g,f) = N(y|f/pos(g+mu0), k/pos(g + mu0)^2) N(g|0, Kg) N(f|0,Kf). Then you sample 
%from the posterior p(f,g|y)
%
% samples =  dgpMCMC(X, y, Kf, Kg, mu0, k, ops)
%    Inputs:
%       X: n x D input data
%       y: n x 1 output data
%       Kf: n x n signal GP kernel matrix      
%       Kg: n x n noise GP kernel matrix
%       mu0: scalar mean of the noise GP
%       k: scalar constant scaling the noise
%       ops: burnin and sampling iterations 
%    Outputs:
%       samples: collected samples for the g vector 
%
%Testing mode: Its takes the samples from the "training" phase together
%with the kernel matrices in the test data and outputs the prediction for the 
%mean, variance and for the (non-Gaussian) predictive distribution 
%for the test outputs.
%
% [mustar varstar, preddstar, samples] =  dgpMCMC(X, y, Kf, Kg, mu0, ops, samples, Kfss, Kgss, Kfstar, Kgstar, ystar)
%    Inputs:
%       X, y, Kf, Kg, mu0: the same as in the training mode 
%       ops: Not used (you can just put empty or any value)
%       samples: the output varable from the training phase
%       Kfss: ntest-dim vector with kf(x_*,x_*) for any test input x_*
%       Kgss: ntest-dim vector with kg(x_*,x_*) for any test input x_*
%       Kfstar: (n x ntest) kernel matrix between test and training inputs
%       Kgstar: (n x ntest) kernel matrix between test and training inputs
%       ystar: actual test output data (optional)
%    Outputs:
%       mustar: (1 x ntest) vector with the predictive means
%       varstar: (1 x ntest) vector with the predictive variances
%       preddstar: (1 x ntest) vector with the predictive density values 
%                    (computed only if ystar is provided)
%       samples: the training samples variable updated to contain also
%       the samples needed to for gstar (the test noisy GP  values). These
%       new samples are needed to compute the predtictive density 

% Author: Luis Muñoz-González
% Based on Michalis Titsias and Miguel Lázaro-Gredilla Heteroscedastic GP
% with MCMC implementation

[n D] = size(X);
k_matrix = k.*eye(n);
meanp=mean(y(:));
y = y - meanp;

% compute Cholesky decomposition for Kg
%jit = 10e-7*mean(diag(Kg));
K_total = [Kf, zeros(n,n);zeros(n,n) Kg];
L_total = chol(K_total);

% training mode
if nargin == 7 
%
%
% initialize g to zero 
fg = zeros(1,2*n);

% compute old log likelihood
const = - (n/2)*log(2*pi);
g2 = fg(n+1:end) + mu0; %Add the mean mu0
g2(g2 < 0) = 0;  %Take the positive part
g_term = sum(log(g2));
L = chol(k_matrix);
L = L';
invL = L\eye(n);     
oldLogLik = g_term + const - sum(log(diag(L)));
y2 = y.*g2';
temp = invL*(-y2(:) + fg(1:n)'); 
oldLogLik = oldLogLik - 0.5*temp'*temp;


% keep space for the samples
BurnIn = ops.BurnIn;
T = ops.T;
StoreEvery = ops.StoreEvery;
num_stored = floor(T/StoreEvery);
samples.fg = zeros(num_stored, 2*n);
samples.LogL = zeros(num_stored, n);
% number of likelihod calls
samples.num_calls =0; 
cnt = 0;
angle_range = 0;


for it = 1:(BurnIn + T) 
%
    % the code here follows Iain Murray's implementation   
    nu = randn(1, 2*n)*L_total;
    hh = log(rand) + oldLogLik;
    if angle_range <= 0
       % Bracket whole ellipse with both edges at first proposed point
       phi = rand*2*pi;
       phi_min = phi - 2*pi;
       phi_max = phi;
    else
       % Randomly center bracket on current point
       phi_min = -angle_range*rand;
       phi_max = phi_min + angle_range;
       phi = rand*(phi_max - phi_min) + phi_min;
    end

    % Slice sampling loop
    num_calls=0;
    while true
       % Compute xx for proposed angle difference and check if it's on the slice
       fgnew = cos(phi)*fg + nu*sin(phi);
       % update the log likelihood
       g2new = fgnew(n+1:end) + mu0; %Add the mean mu0
       g2new(g2new < 0) = 0;  %Take the positive part
       g_term = sum(log(g2new));    
       oldLogLik = g_term + const - sum(log(diag(L)));
       y2 = y.*g2new';
       temp = invL*(-y2(:) + fgnew(1:n)'); 
       oldLogLik = oldLogLik - 0.5*temp'*temp;
       % end updating log likelihood 
       num_calls = num_calls + 1; 
       if oldLogLik > hh
           % New point is on slice, ** EXIT LOOP **
           break;
       end
       % Shrink slice to rejected point
       if phi > 0
           phi_max = phi;
       elseif phi < 0
           phi_min = phi;
       else
           error('BUG DETECTED: Shrunk to current position and still not acceptable.');
       end
       % Propose new angle difference
       phi = rand*(phi_max - phi_min) + phi_min;
    end
    fg = fgnew;
     
    samples.num_calls = samples.num_calls + num_calls;
    % keep samples after burn in
    if (it > BurnIn) & (mod(it,StoreEvery) == 0)
    %
        cnt = cnt + 1;
        samples.fg(cnt,:) = fg;
        samples.LogL(cnt) = oldLogLik;   
    %
    end
    %        
   out1 = samples; 
end


% TESTING MODE 
else
 ntest = size(Kfstar,2);   
 out1 = zeros(ntest,1);
 out2 = zeros(ntest,1);
 out3 = zeros(ntest,1);
 out4 = zeros(ntest,1);
 T = size(samples.fg,1);
 Lf = chol(Kf);
 Lg = chol(Kg);
 % iterate across all training samples 
 for j=1:T
    v = Lf'\Kfstar;
    varf = Kfss(:) - sum(v.*v)';
    out2 = out2 + varf;
    muf = Kfstar'*(Lf\(Lf'\(samples.fg(j,1:n)'))); 
    out1 = out1 + muf; 
    v = Lg'\Kgstar;
    varg = Kgss(:) - sum(v.*v)';
    out4 = out4 + varg;
    mug = Kgstar'*(Lg\(Lg'\(samples.fg(j,n+1:end)')));   
    out3 = out3 + mug;
    fstar = muf(:) + randn(ntest,1).*sqrt(abs(varf(:)));
    samples.fstar(j,:) = fstar; 
    gstar = mug(:) + randn(ntest,1).*sqrt(abs(varg(:)));
    samples.gstar(j,:) = gstar; 
        
    
 end

 out1 = out1/T;  
 %out2 = out2/T;
 out3 = out3/T;
 %out4 = out4/T;
 out2 = var(samples.fstar)';
 out4 = var(samples.gstar)';
end

end