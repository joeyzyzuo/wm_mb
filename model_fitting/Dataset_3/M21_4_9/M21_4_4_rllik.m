function LL = M21_4_4_rllik(x,subdata,opts)

% Likelihood function for Daw two-step paradigm in Kool, Cushman, &
% Gershman (2016).
%
% Depending on opts, the function reflects either a full hybrid model (opts.model = 1),
% a fully model-based model (opts.model = 2), or a model-free model
% (opts.model = 3). The fields opts.st and opts.respst determine the
% inclusion of stickiness parameters.
%
% Wouter Kool, Aug 2016 (based on code by Samuel J. Gershman)

y = zeros(1,9);
y(opts.ix==1) = x;

% parameters
b_1 = y(1);           % softmax inverse temperature
lr_1 = y(2);          % learning rate
lambda = y(3);      % eligibility trace decay
w = y(4);           % mixing weight
st = y(5);          % stickiness
lr_2 = y(6);           % softmax inverse temperature
b_2 = y(7);          % learning rate
a_weight = y(8);   
respst = y(9);
% initialization
Qd = zeros(3,2);            % Q(s,a): state-action value function for Q-learning

W2 = zeros(2,2); 

Tm = [.5 .5; .5 .5];        % transition matrix
M = [0; 0];                 % last choice structure
R = [0; 0];                 % last response structure

counts = zeros(2,2);        % counting transitions

N = length(subdata.choice1);

LL = 0;

% loop through trials
for t = 1:N

%     Break if trial was missed
    if (subdata.choice1(t) == -1 || subdata.choice2(t) == -1)
        continue
    end
    
    state2 = subdata.state2(t)+1;
    
    if subdata.stim_1_left(t) == 2
        R = flipud(R);                                                          % arrange R to reflect stimulus mapping
    end
    
    maxQ = max(Qd(2:3,:),[],2);                                                 % optimal reward at second step
    Qm = Tm'*maxQ;                                                              % compute model-based value function
    
    
    maxW2 = max(W2,[],2); 
    W1 = Tm'*maxW2;
    
    Q = w*W1 + (1-w)*Qd(1,:)'+ st.*M + respst.*R;                                  % mix TD and model-based values

    LL = LL + b_1*Q(subdata.choice1(t))-logsumexp(b_1*Q);                           % update likelihoods
    logP1=b_1*Q(subdata.choice1(t))-logsumexp(b_1*Q);
    
    
    Q2 = w*W2(subdata.state2(t),:)' + (1-w)*Qd(state2,:)';
    logP2=b_2*Q2(subdata.choice2(t)) - logsumexp(b_2*Q2(:));
    LL = LL + b_2*Q2(subdata.choice2(t)) - logsumexp(b_2*Q2(:));
%     LL = LL + b_2*Q2(state2,subdata.choice2(t)) - logsumexp(b_2*Q2(state2,:));

    M = [0; 0];
    M(subdata.choice1(t)) = 1;                                                  % make the last choice sticky
    
    R = zeros(2,1);
    if subdata.choice1(t) == subdata.stim_1_left(t)
        R(1) = 1;                                                               % make the last response sticky
    else
        R(2) = 1;
    end
    
    dtQ(1) = Qd(state2,subdata.choice2(t)) - Qd(1,subdata.choice1(t));          % backup with actual choice (i.e., sarsa)
    
    
    if dtQ(1)>2
        dsadsa=2;
    end
    
    Qd(1,subdata.choice1(t)) = Qd(1,subdata.choice1(t)) + lr_1*dtQ(1);            % update TD value function
     
    dtQ(2) = subdata.win(t) - Qd(state2,subdata.choice2(t));                    % prediction error (2nd choice)
    
    if dtQ(2)>2
        dsadsa=2;
    end
    Qd(state2,subdata.choice2(t)) = Qd(state2,subdata.choice2(t)) + lr_2*dtQ(2);  % update TD value function
    Qd(1,subdata.choice1(t)) = Qd(1,subdata.choice1(t)) + lambda*lr_1*dtQ(2);     % eligibility trace
    
%     fai=0;
    if t <= 4
        reward_4 = subdata.win(1:t);
    else
        reward_4 = subdata.win(t-3:t);
    end
    
    weight1_list = flip([1]);
    weight2_list = flip([a_weight,1-a_weight]);
    weight3_list = flip([a_weight,(1-a_weight)*a_weight,(1-a_weight)*(1-a_weight)]);
    weight4_list = flip([a_weight,(1-a_weight)*a_weight,(1-a_weight)*(1-a_weight)*a_weight,(1-a_weight)*(1-a_weight)*(1-a_weight)]);
    
    if t <= 4
        index=find((subdata.state2(1:t)==1) & (subdata.choice2(1:t)==1));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(1,1) = wm_value;
        end
        index=find((subdata.state2(1:t)==1) & (subdata.choice2(1:t)==2));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(1,2) = wm_value;
        end
        index=find((subdata.state2(1:t)==2) & (subdata.choice2(1:t)==1));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(1,2) = wm_value;
        end
        index=find((subdata.state2(1:t)==2) & (subdata.choice2(1:t)==2));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(2,2) = wm_value;
        end
    else
        index=find((subdata.state2(t-3:t)==1) & (subdata.choice2(t-3:t)==1));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(1,1) = wm_value;
        end
        index=find((subdata.state2(t-3:t)==1) & (subdata.choice2(t-3:t)==2));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(1,2) = wm_value;
        end
        index=find((subdata.state2(t-3:t)==2) & (subdata.choice2(t-3:t)==1));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(2,1) = wm_value;
        end
        index=find((subdata.state2(t-3:t)==2) & (subdata.choice2(t-3:t)==2));
        if isnan(mean(reward_4(index)))
            continue
        else
            switch length(reward_4(index)')
                case 1   
                    wm_value = sum(weight1_list.* reward_4(index)');
                case 2   
                    wm_value = sum(weight2_list.* reward_4(index)');
                case 3   
                    wm_value = sum(weight3_list.* reward_4(index)');
                case 4   
                    wm_value = sum(weight4_list.* reward_4(index)');
            end
            W2(2,2) = wm_value;
        end
    end
    
    % pick the most likely transition matrix
    counts(subdata.state2(t),subdata.choice1(t)) = counts(subdata.state2(t),subdata.choice1(t))+1;
    
    if sum(diag(counts))>sum(diag(rot90(counts)))
        Tm = [.7 .3; .3 .7];        % transition matrix
    end
    if sum(diag(counts))<sum(diag(rot90(counts)))
        Tm = [.3 .7; .7 .3];        % transition matrix
    end
    if sum(diag(counts))==sum(diag(rot90(counts)))
        Tm = [.5 .5; .5 .5];        % transition matrix
    end
end

end
