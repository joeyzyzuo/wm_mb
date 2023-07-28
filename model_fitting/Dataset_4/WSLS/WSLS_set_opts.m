function [opts, param] = WSLS_set_opts(opts)

% Code that sets up different options, and empirical priors for model-fitting
% procedure of novel two-step paradigm in Kool, Cushman, & Gershman (2016).
% Parameters of the prior are chosen after Gershman (2016).
%
% Wouter Kool, Aug 2016

opts.ix = ones(1,1);

param(1).name = 'epi';
param(1).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(1).lb = 0;
param(1).ub = 1;

param = param(opts.ix==1);

end