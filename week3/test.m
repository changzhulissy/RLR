%% initialization
episode_num = 500;
test_num = 500;
torque_min = -5;
torque_max = 5;
pos_min = -pi;
pos_max = pi;
%% 
acc = dynamics(vel,pos,torque);%0.1s action update
[new_pos,new_vel] = euler(pos, vel,acc);% 0.001s state update
%% Q-Learning with variable resolution function
Q = 0;
for i = 1:episode_num
    pos = pi;
    vel = 0;
    while(1)
        % select action a according to exploration-exploitation strategy  
        torque = variable_resolution;
        acc = dynamics(vel,pos,torque);
        [pos,vel] = euler(pos, vel,acc);
        r = reward(pos,vel);
        % estimate maximum Qmax
        Qmax = ;
        Q(pos,torque) = r+gamma*Qmax;
        if pos == 0
            break
        end
    end
    % test 
    for j = 1:test_num
        R_test[i,j] =; 
    end
end
%% plot acc_reward