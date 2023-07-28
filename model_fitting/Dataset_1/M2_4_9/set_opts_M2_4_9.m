% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
function [opts, param] = set_opts_M2_4_9(opts)
% Original code by Wouter Kool (Kool, Cushman, & Gershman, 2016),Florian
% Bolenz (Bolenz, Kool, Reiter, & Eppinger, 2019)
% Code adapted by Joey Zuo

opts.ix = ones(1,13);

if ~opts.lambda, opts.ix(3) = 0; end
if ~opts.eta, opts.ix(4) = 0; end
if ~opts.kappa, opts.ix(5) = 0; end
if ~opts.fai, opts.ix(10) = 0; end
if ~opts.st, opts.ix(11) = 0; end
if ~opts.respst, opts.ix(12) = 0; end

% create parameter structure
g = [3 0.2];
param(1).name = 'inverse temperature';
param(1).logpdf = @(x) sum(log(gampdf(x,g(1),g(2))));  % log density function for prior
param(1).lb = 0;   % lower bound
param(1).ub = 5;  % upper bound

param(2).name = 'reward learning rate';
param(2).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(2).lb = 0;
param(2).ub = 1;

param(3).name = 'eligibility trace decay';
param(3).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(3).lb = 0;
param(3).ub = 1;

param(4).name = 'transition learning rate';
param(4).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(4).lb = 0;
param(4).ub = 1;

param(5).name = 'counterfactual transition learning rate';
param(5).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(5).lb = 0;
param(5).ub = 1;

param(6).name = 'mixing weight low stable';
param(6).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(6).lb = 0;
param(6).ub = 1;

param(7).name = 'mixing weight high stable';
param(7).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(7).lb = 0;
param(7).ub = 1;

param(8).name = 'mixing weight low variable';
param(8).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(8).lb = 0;
param(8).ub = 1;

param(9).name = 'mixing weight high variable';
param(9).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(9).lb = 0;
param(9).ub = 1;

param(10).name = 'fai_decay_wm';
param(10).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(10).lb = 0;
param(10).ub = 1;

mu = 0; sd = 1;   % parameters of choice stickiness
param(11).name = 'choice stickiness';
param(11).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(11).lb = -4;
param(11).ub = 4;

mu = 0; sd = 1;    % parameters of response stickiness
param(12).name = 'response stickiness';
param(12).logpdf = @(x) sum(log(normpdf(x,mu,sd)));
param(12).lb = -4;
param(12).ub = 4;

param(13).name = 'mixing_weight_base';
param(13).logpdf = @(x) sum(log(betapdf(x,2,2)));
param(13).lb = 0;
param(13).ub = 1;

param = param(opts.ix==1);

end
