% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
function  [LL, dtQ] = M0_0_rllik(x,subdata,opts)
% Original code by Wouter Kool (Kool, Cushman, & Gershman, 2016),Florian
% Bolenz (Bolenz, Kool, Reiter, & Eppinger, 2019)
% Code adapted by Joey Zuo

y = zeros(1,11);
y(opts.ix==1) = x;

switch opts.model
    case 2
        % for pure model-based learner, set all mixing weights to 1
        y(6:9) = 1;
    case 3
        % for pure model-free learner, set all mixing weights to 0
        y(6:9) = 0;
end
if ~opts.lambda, y(3) = 0; end
if ~opts.eta, y(4) = 1; end
if ~opts.kappa, y(5) = y(4); end % countfactual
if ~opts.st, y(10) = 0; end
if ~opts.respst, y(11) = 0; end

% parameters
b = y(1);                   % softmax inverse temperature
lr = y(2);                  % reward learning rate
lambda = y(3);              % eligibility trace decay
eta_variable = y(4);        % transition learning rate
kappa_variable = y(5);      % counterfactual transition learning rate
w_low_stable = y(6);        % mixing weight low stable
w_high_stable = y(7);       % mixing weight high stable
w_low_variable = y(8);      % mixing weight low variable
w_high_variable = y(9);     % mixing weight high variable
st = y(10);                 % stimulus stickiness
respst = y(11);             % response stickiness

% initialization
Qmf = ones(2,2)*4.5;        % Q(s,a): First-stage state-action values
Q2 = ones(2,1)*4.5;         % Q(s,a): Second-stage state-action values
Tm = cell(2,1);
Tm{1,:} = [.5 .5; .5 .5];   % transition matrix s1=1
Tm{2,:} = [.5 .5; .5 .5];   % transition matrix s1=2
M = [0 0; 0 0];             % last choice structure ,last choice sticky
R = [0; 0];                 % last choice structure ,response sticky
LL = 0;                     % log-likelihood
N = size(subdata.choice,1); % number of trials

% loop through trials
for t = 1:N 
    
    if subdata.timeout(t,1) == 1    % skip trial if timed out at first stage
        continue
    end
    
    if (subdata.stimuli(t,1) == 2) || (subdata.stimuli(t,1) == 4)
        R = flipud(R);              % arrange R to reflect stimulus mapping
    end
    
    s1 = subdata.s(t,1);            % first-stage state
    s2 = subdata.s(t,2);            % second-stage state
    a = mod(subdata.choice(t)-1,2)+1;   % action (code action 3 as 1, and action 4 as 2)

    Qmb = Tm{s1}'*Q2;                % compute model-based value function
    
    % set mixing weight and transition learning rates according to
    % experimental condition
    if subdata.stake(t) == 1 && subdata.blockCondition(t) == 0
        w = w_low_stable;
        eta = 1;
        kappa = 1;
    elseif subdata.stake(t) == 5 && subdata.blockCondition(t) == 0
        w = w_high_stable;
        eta = 1;
        kappa = 1;
    elseif subdata.stake(t) == 1 && subdata.blockCondition(t) == 1
        w = w_low_variable;
        eta = eta_variable;
        kappa = kappa_variable;
    elseif subdata.stake(t) == 5 && subdata.blockCondition(t) == 1
        w = w_high_variable;
        eta = eta_variable;
        kappa = kappa_variable;
    else
        error('Could not determine model-based weight')
    end

    Q = w*Qmb + (1-w)*Qmf(s1,:)' + st.*M(s1,:)' + respst.*R;        % mix TD and model-based value
    
    LL = LL + b*Q(a) - logsumexp(b*Q);                              % update likelihoods
    
    M = zeros(2,2);
    M(s1,a) = 1;                                                    % make the last choice sticky
    
    R = zeros(2,1);
    if a+(s1-1)*2 == subdata.stimuli(t,1)
        R(1) = 1;                                                   % make the last response sticky
    elseif a+(s1-1)*2 == subdata.stimuli(t,2)
        R(2) = 1;
    else
        error('Error')
    end
    
    % update transition matrix
    spe = 1 - Tm{s1,:}(s2, a);                              % state prediction error
    Tm{s1}(s2, a) = Tm{s1}(s2, a) + eta*spe;                % update observed transition
    Tm{s1}(abs(s2-3), a) = Tm{s1}(abs(s2-3), a)*(1-eta);    % reduce alternative transition probability
    
    cf_spe = 1 - Tm{s1}(abs(s2-3), abs(a-3));               % counterfactual state prediction error
    Tm{s1}(abs(s2-3), abs(a-3)) =  Tm{s1}(abs(s2-3), abs(a-3)) + kappa*cf_spe;  % update transition
    Tm{s1}(s2, abs(a-3)) = Tm{s1}(s2, abs(a-3))*(1-kappa);  % reduce transition
    
    % update state-action values
    dtQ(1) = Q2(s2) - Qmf(s1,a);                                    % backup with actual choice (i.e., sarsa)
    Qmf(s1,a) = Qmf(s1,a) + lr*dtQ(1);                              % update TD value function
    
    if subdata.timeout(t,2) == 1              
        continue % do not update second-stage Q values if timed out at second stage
    end
    
    dtQ(2) = subdata.points(t) - Q2(s2);                            % prediction error (2nd choice)
    
    Q2(s2) = Q2(s2) + lr*dtQ(2);                                    % update TD value function
    Qmf(s1,a) = Qmf(s1,a) + lambda*lr*dtQ(2);                       % eligibility trace

end