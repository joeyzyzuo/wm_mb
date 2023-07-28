% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
function  [LL, dtQ] = M3_4_9_rllik(x,subdata,opts)
% Original code by Wouter Kool (Kool, Cushman, & Gershman, 2016),Florian
% Bolenz (Bolenz, Kool, Reiter, & Eppinger, 2019)
% Code adapted by Joey Zuo

y = zeros(1,13);
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
if ~opts.kappa, y(5) = y(4); end
if ~opts.fai, y(10) = 0; end
if ~opts.st, y(11) = 0; end
if ~opts.respst, y(12) = 0; end

% parameters
b = y(1);                   % softmax inverse temperature
lr = y(2);                  % reward learning rate
lambda = y(3);              % eligibility trace decay
eta_variable = y(4);        % transition learning rate
kappa_variable = y(5);      % counterfactual transition learning rate
w_WM_low_stable = y(6);        % mixing weight RL_low_stake
w_WM_high_stable = y(7);       % mixing weight RL_high_stake
w_WM_low_variable = y(8);      % mixing weight WM_stable
w_WM_high_variable = y(9);     % mixing weight WM_variable
decay_fai = y(10);             % wm decay
st = y(11);                 % stimulus stickiness
respst = y(12);             % response stickiness
mixing_weight_base = y(13);

% initialization
Qmf = ones(2,2)*4.5;        % Q(s,a): First-stage state-action values
Q2 = ones(2,1)*4.5;         % Q(s,a): Second-stage state-action values

W2 = ones(2,1)*4.5;         % Q(s,a): Second-stage WM values

Tm = cell(2,1);
Tm{1,:} = [.5 .5; .5 .5];   % transition matrix s1=1
Tm{2,:} = [.5 .5; .5 .5];   % transition matrix s1=2
M = [0 0; 0 0];             % last choice structure ,last choice sticky
R = [0; 0];                 % last choice structure ,botton sticky

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

    
    WQ2 = mixing_weight_base * W2 + (1 - mixing_weight_base) * Q2;
    
    W = Tm{s1}'*WQ2;
%     Qmb = Tm{s1}'*Q2;               % compute model-based value function
%     
%     W = Tm{s1}'*W2;               % compute model-based value function
    
    
    if subdata.stake(t) == 1 && subdata.blockCondition(t) == 0
        wwm = w_WM_low_stable;
        eta = 1;
        kappa = 1;
    elseif subdata.stake(t) == 5 && subdata.blockCondition(t) == 0
        wwm = w_WM_high_stable;
        eta = 1;
        kappa = 1;
    elseif subdata.stake(t) == 1 && subdata.blockCondition(t) == 1
        wwm = w_WM_low_variable;
        eta = eta_variable;
        kappa = kappa_variable;
    elseif subdata.stake(t) == 5 && subdata.blockCondition(t) == 1
        wwm = w_WM_high_variable;
        eta = eta_variable;
        kappa = kappa_variable;
    else
        error('Could not determine model-based weight')
    end

    Q = wwm*W + (1-wwm)*Qmf(s1,:)' + st.*M(s1,:)' + respst.*R;        % mix TD and model-based value
    
    LL = LL + b*Q(a) - logsumexp(b*Q); 

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
    
    % refresh wm
    W2(s2) = subdata.points(t) ; 

    % wm decay

    if s2==1
        W2(2) = W2(2) + decay_fai * (4.5-W2(2)); 
    else
        W2(1) = W2(1) + decay_fai * (4.5-W2(1));
    end
                
end
end