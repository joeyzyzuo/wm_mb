% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
function [opts, param] = set_opts_wsls(opts)
% Original code by Wouter Kool (Kool, Cushman, & Gershman, 2016),Florian
% Bolenz (Bolenz, Kool, Reiter, & Eppinger, 2019)
% Code adapted by Joey Zuo

opts.ix = ones(1,1);

param(1).name = 'epi';
param(1).logpdf = @(x) sum(log(betapdf(x,1.1,1.1)));
param(1).lb = 0;
param(1).ub = 1;

param = param(opts.ix==1);

end
