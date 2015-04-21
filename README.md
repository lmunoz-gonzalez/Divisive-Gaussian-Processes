# Divisive-Gaussian-Processes
Code to perform approximate inference on a Divisive Gaussian Process model with Expectation Propagation and the Laplace approximation


This package contains the implementation of the methods described in the papers: "Divisive Gaussian Processes for Nonstationary Regression" and "Laplace Approximation for Divisive Gaussian Processes for Nonsationary Regression" (both authored by Luis Muñoz-González, Miguel Lázaro-Gredilla, and Aníbal R. Figueiras-Vidal). Please, if you use this software, reference the following papers as:
- L. Muñoz-González, M. Lázaro-Gredilla, and A.R. Figueiras-Vidal, "Divisive Gaussian Processes for Nonstationary Regression", IEEE Transactions on Neural Networks and Learning Systems, vol. 25, no. 11, pp. 1991-2003, 2014.
- L. Muñoz-González, M. Lázaro-Gredilla, and A.R. Figueiras-Vidal, "Laplace Approximation for Divisive Gaussian Processes for Nonsationary Regression", IEEE Transactions on Pattern Analysis and Machine Intelligence. Under review, 2015.


CONSIDERATIONS
-----------------------------------------
This package has been developed using code from Carl Rasmussen's GPML Matlab implementation 
http://www.gaussianprocess.org/gpml/code/matlab/doc/


USAGE
-----------------------------------------

- To get started with the package run the script demoDGP.m.
- It is possible to modify the function DGP_ui to use different covariance functions. We use an isometric exponential covariance function as default (although it is possible to use ARD exponential covariance functions removing comments in the code where indicated).
- Datasets folder contains one random split from each data set used for the experiments in both papers.
