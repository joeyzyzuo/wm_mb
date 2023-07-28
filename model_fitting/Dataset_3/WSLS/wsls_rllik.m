% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
function  [LL, dtQ] = wsls_rllik(x,subdata,opts)
% Original code by Wouter Kool (Kool, Cushman, & Gershman, 2016),Florian
% Bolenz (Bolenz, Kool, Reiter, & Eppinger, 2019)
% Code adapted by Joey Zuo

y = zeros(1,1);
y(opts.ix==1) = x;
dtQ=0;

% parameters
epi = y(1);                   % softmax inverse temperature

% initialization
Tm = [.5 .5; .5 .5];        % transition matrix
LL = 0;                     % log-likelihood
N = length(subdata.choice1);
subdata.target = (subdata.state2 - 1) * 2 + subdata.choice2;
% loop through trials
for t = 1:N 
    
    if (subdata.choice1(t) == -1 || subdata.choice2(t) == -1)
        continue
    end
    
    if t==1
        P = 0.5;
        LL = LL + log(P); 
        continue
    end
    
    if (subdata.choice1(t-1) == -1 || subdata.choice2(t-1) == -1)
        P = 0.5;
        LL = LL + log(P); 
        continue
    end
    
    % step1
    if subdata.win(t-1)
        if subdata.choice1(t) == subdata.state2(t-1)
            P = 1 - epi/2;
        else
            P = epi/2;
        end
    else
        if subdata.choice1(t) ~= subdata.state2(t-1)
            P = 1 - epi/2;
        else
            P = epi/2;
        end
    end
    
    LL = LL + log(P); 
    
    % step2
    % 如果同一个s2
    if subdata.state2(t) == subdata.state2(t-1)
        if subdata.win(t-1)
            if subdata.target(t) == subdata.target(t-1)
                P = 1 - epi/2;
            else
                P = epi/2;
            end
        else
            if subdata.target(t) ~= subdata.target(t-1)
                P = 1 - epi/2;
            else
                P = epi/2;
            end
        end
    else% 如果不同的s2
        P = 1/2;
    end

    LL = LL + log(P); 

end     
end