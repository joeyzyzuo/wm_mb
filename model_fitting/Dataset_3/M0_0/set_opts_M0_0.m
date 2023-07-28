function [opts, param] = set_opts_M0_0(opts)

% Code that sets up different options, and empirical priors for model-fitting
% procedure for the Daw two-step paradigm in Kool, Cushman, & Gershman (2016).
% Parameters of the prior are chosen after Gershman (2016).
%
% Wouter Kool, Aug 2016

opts.ix = ones(1,8);

g = [1.2 5];
param(1).name = 'inverse temperature_1';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;   % lower bound
param(1).ub = 5;  % upper bound

param(2).name = 'reward learning rate_1';
param(2).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(2).lb = 0;
param(2).ub = 1;

param(3).name = 'eligibility trace decay';
param(3).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(3).lb = 0;
param(3).ub = 1;

param(4).name = 'mixing weight working memory';
param(4).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(4).lb = 0;
param(4).ub = 1;

mu = 0; sd = 1;   % parameters of choice stickiness
param(5).name = 'choice stickiness';
param(5).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(5).lb = -2;
param(5).ub = 2;

param(6).name = 'reward learning rate_2';
param(6).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(6).lb = 0;
param(6).ub = 1;

g = [1.2 5];
param(7).name = 'inverse temperature_2';
param(7).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(7).lb = 0;   % lower bound
param(7).ub = 5;  % upper bound

mu = 0; sd = 1;   % parameters of choice stickiness
param(8).name = 'response stickiness';
param(8).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(8).lb = -2;
param(8).ub = 2;
param = param(opts.ix==1);

end
