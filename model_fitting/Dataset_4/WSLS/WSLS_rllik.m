function LL = WSLS_rllik(x,subdata,opts)
% 选择M的那个参数，因为daw2011里说The indicator function 
%  ( ) is defined as 1 if a is a top-stage action and is the same one as was chosen on the previous trial, 
% zero otherwise.
% 也就是action.

y = zeros(1,1);
y(opts.ix==1) = x;

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
