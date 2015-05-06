function [L] = chol_tridiag(M);
%Cholesky factorization of a block diagonal matrix
n3 = length(M);
n = n3/3;

A = M(1:n);
B = M(n+1:2*n);
C = M(2*n+1:end);

P = sqrt(A);
Q = C./P;
R = sqrt(B - Q.^2);
L = [diag(P) diag(Q);zeros(n) diag(R)];