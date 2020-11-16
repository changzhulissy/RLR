%% initialization

a = [1,2,3,4]';
%% SARSA
gamma = 0.9;
alpha = 0.01;
epsilon = 0.1;
Q = rand(12,4);
Q(10,:) = 0;
episodes_num = 1000;
for i = 1:episodes_num
    state = 1;
    if rand<epsilon
            action = randi(length(a));
    else        
            [~,action] = max(Q(state,:));
    end
    while(1)
        [snext,r] = simulator(state,action);
        if rand<epsilon
            action_next = randi(length(a));
        else        
            [~,action_next] = max(Q(snext,:));
        end
        Q(state,action) = Q(state,action)+alpha*(r+gamma*Q(snext,action_next)-Q(state,action));
        state = snext;
        action = action_next;
        if snext == 10
            break
        end
    end

end
[~,policy]= max(Q,[],2)
%% Q-Learning
epsilon = 0.3;
gamma = 0.9;
alpha = 0.2;%step size
Q = rand(12,4);
Q(10,:) = 0;
episodes_num = 50;
for i = 1:episodes_num
    % set the initial observation S
    state = 1;
    while(1)
    %a
        if rand<epsilon
            action = randi(length(a));
        else        
            [~,action] = max(Q(state,:));
        end
        %b
        [snext,r] = simulator(state,action);
        %c
        Q(state,action) = Q(state,action)+alpha*(r+gamma*max(Q(snext,:))-Q(state,action));
        state = snext;
        if state == 10
            break
        end
    end
end
[~,policy]= max(Q,[],2)