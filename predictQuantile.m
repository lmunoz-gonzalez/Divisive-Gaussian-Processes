function [y] = predictQuantile(mu_f,var_f,mu_g,var_g,k,Q)

n = length(mu_f);
y = zeros(n,1);

for i=1:n        
        %Initial approximation
        m = mu_f(i)/mu_g(i);
        Cnorm = normcdf(0,mu_g(i),sqrt(var_g(i)));
        sigma = sqrt(var_g(i)*Q^2 + k + var_f(i))/mu_g(i);
        initial_approx = norminv(Q,m,sigma);        
        
        delta = 4*max([sqrt(var_f(i)) sqrt(var_g(i))]);
        %initial_approx = mu_f(i)./mu_g(i);
        
        %Initial lower bound
        a = initial_approx - delta; %lower bound
        [ya] = evalCDF(mu_f(i),var_f(i),mu_g(i),var_g(i),k,Cnorm,Q,a);
        iter = 0;
        while(ya >= 0)
            iter = iter + 1;
            a = a - 0.5*exp(iter);
            [ya] = evalCDF(mu_f(i),var_f(i),mu_g(i),var_g(i),k,Cnorm,Q,a);
        end
                    
        %Initial upper bound
        b = initial_approx + delta; %upper bound
        [yb] = evalCDF(mu_f(i),var_f(i),mu_g(i),var_g(i),k,Cnorm,Q,b);
        iter = 0;
        while (yb <= 0)
            iter = iter + 1;
            b = b + 0.5*exp(iter);
            [yb] = evalCDF(mu_f(i),var_f(i),mu_g(i),var_g(i),k,Cnorm,Q,b);
        end

        
       %Parameters of bisection method 
        tol = 1e-4;  %Tolerance for bisection method
        max_iter = 50; %Maximum number of iterations
        iter = 0; %Initial iteration
        dif = 1e10; %Initial difference
        %Iterate
        while ((abs(dif) > tol) && (iter < max_iter))
            iter = iter + 1;
            c =(a+b)/2;
            %Evaluate function in c
            [yc] = evalCDF(mu_f(i),var_f(i),mu_g(i),var_g(i),k,Cnorm,Q,c);
            if (abs(yc) < tol)
                a = c;
                b = c;
            elseif (yb*yc > 0)
                b = c;
                yb = yc;
            else
                a = c;
                ya = yc;
            end
            dif = b-a;
        end
        if (iter == max_iter)
            disp('WARNING: BISECTION METHOD HAS NOT CONVERGED');
        end
        y(i) = (a+b)/2;        
%     end
end


end

function [y] = evalCDF(mu_f,var_f,mu_g,var_g,k,Cnorm,Q,x)
    denom = sqrt(var_g*(x^2) + k + var_f);
    x_2 = [(mu_g*x - mu_f)/denom; mu_g/sqrt(var_g)];
    gamma = sqrt(var_g)*x/denom;
    sigma_x = [1 gamma; gamma 1];
    y = Cnorm + mvncdf(x_2, [0; 0], sigma_x) - Q;
end