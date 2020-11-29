%% initialization
episode_num = 100;
iter_num = 500;
test_num = 500;
a_min = -5;
a_max = 5;
a_interval = a_max-a_min;

vel_max = 2*pi;
vel_min = -2*pi;

pos_min = -pi;
pos_max = pi;
select_seg_a = 20;
alpha = 0.01;%forget_rate
gamma = 0.85;%discount_rate
thr_error = 1;
%% visualization
%vars   = [omega_0      theta_0      theta_t0];
%values = [omega_0Value theta_0Value theta_t0Value];
%thetaSolPlot = subs(thetaSol,vars,values);
[new_pos, new_vel, pos_mat] = euler_pend(5*pi/6,0,0,0.001*30);

x_pos = sin(pos_mat);
y_pos = -cos(pos_mat);

fanimator(@fplot,x_pos,y_pos,'ko','MarkerFaceColor','k','AnimationRange',[0 5]);
%fanimator(@(t) plot([0 x_pos(t)],[0 y_pos(t)],'k-'),'AnimationRange',[0 5]);
%fanimator(@(t) text(-0.3,0.3,"Timer: "+num2str(t,2)+" s"),'AnimationRange',[0 5]);
%[new_pos,new_vel] = euler(pos, vel,acc);% 0.001s state update
%% Q-Learning with variable resolution function
pos_segnum = 1; % initial segmentation of position (state1)
pos_nodes = [pos_min,pos_max];
vel_segnum = 1; % initial segmentation of velocity (state2)
vel_nodes = [vel_min,vel_max];
a_segnum = 1; % initial segmentation of torque (action)
a_nodes = [a_min,a_max];

Q = zeros(pos_segnum,vel_segnum,a_segnum); % initialization Q_mean statistic
S = ones(pos_segnum,vel_segnum,a_segnum); % initialization S variance statisctic
n = ones(pos_segnum,vel_segnum,a_segnum); % initial sample number (n)

pos_ten = zeros(episode_num, iter_num);
vel_ten = zeros(episode_num,iter_num);
r_ten = zeros(episode_num,iter_num);
for i = 1:episode_num
    pos = pi;
    vel = 0;
    sample = zeros(iter_num,7);
    for j = 1:iter_num 
        % behavior policy: select action a according to exploration-exploitation strategy 
        for k = 1:pos_segnum
            if pos > pos_nodes(k)
                if pos <= pos_nodes(k+1)
                   pos_i =k;
                end
            end
        end
        for m = 1:vel_segnum
            if vel > vel_nodes(m)
                if vel <= vel_nodes(m+1)
                    vel_i = m;
                end
            end
        end       
        torque = select_a(Q(pos_i,vel_i,:),S(pos_i,vel_i,:),n(pos_i,vel_i,:),select_seg_a,a_nodes,a_segnum,a_min,a_max);
       
        for m = 1:a_segnum
            if torque > a_nodes(m)
                if torque <= a_nodes(m+1)
                    a_i = m;
                end
            end
        end
        [pos_new,vel_new,~] = euler_pend(pos,vel,torque,0.1);
        r = reward(pos_new,vel_new);
        
        % exploration policy: estimate maximum Qmax       
        Q_perm = zeros(select_seg_a,1);
        for kk = 1:select_seg_a
            a_test = a_min+a_interval/select_seg_a*kk;
            for m = 1:a_segnum
                if a_test > a_nodes(m)
                    if a_test <= a_nodes(m+1)
                        a_test_i = m;
                    end
                end
            end
            S_test = S(pos_i,vel_i,a_test_i);
            Q_test = Q(pos_i,vel_i,a_test_i);
            n_test = n(pos_i,vel_i,a_test_i);
            t_test = tinv(0.95,n_test);
            Q_perm(kk)= Q_test+t_test*sqrt(S_test^2/n_test);
        end
        q = r+gamma*max(Q_perm);
        
        % update Q(s,a)
        n(pos_i,vel_i,a_i) = n(pos_i,vel_i,a_i)+1;
        sample(j,:) = [pos,vel,torque,pos_i,vel_i,a_i,q]; % update only in one episode
        S_i = S(pos_i,vel_i,a_i);
        Q_i = Q(pos_i,vel_i,a_i); 
        Q(pos_i,vel_i,a_i) = Q_i+alpha*(q-Q_i);
        S(pos_i,vel_i,a_i) = sqrt(S_i^2+alpha*((q-Q_i)^2-S_i^2));
        if (q-Q(pos_i,vel_i,a_i))^2>=thr_error&&n(pos_i,vel_i,a_i)>10
            pos_size = pos_nodes(pos_i+1)-pos_nodes(pos_i);
            vel_size = vel_nodes(vel_i+1)-vel_nodes(vel_i);
            a_size = a_nodes(a_i+1)-a_nodes(a_i);
            split_dim = max([pos_size/(2*pi),vel_size/(vel_max-vel_min),a_size/a_interval]);
            if pos_size/(2*pi) == split_dim
                q1 = 0;
                q2 = 0;
                pos_segnum = pos_segnum+1;
                pos_nodes = [pos_nodes(1:pos_i),pos_nodes(pos_i)+pos_size/2,pos_nodes((pos_i)+1:end)];
                for kk = 1:j
                    if sample(kk,5)==vel_i
                        if sample(kk,6)==a_i
                            if sample(kk,1)>pos_nodes(pos_i)&&sample(kk,1)<=pos_nodes(pos_i+1)
                                q1 = [q1,sample(kk,7)];
                            elseif sample(kk,1)> pos_nodes(pos_i+1) && sample(kk,1)<= pos_nodes(pos_i+2)
                                q2 = [q2,sample(kk,7)];
                            end
                        end
                    end
                end
                Q(:,vel_i,a_i) = [Q(1:(pos_i-1),vel_i,a_i);mean(q1);mean(q2);Q((pos_i+1):end,vel_i,a_i)];
                S(:,vel_i,a_i) = [S(1:(pos_i-1),vel_i,a_i);std(q1);std(q2);S((pos_i+1):end,vel_i,a_i)];
                n(:,vel_i,a_i) = [n(1:(pos_i-1),vel_i,a_i);length(q1);length(q2);n((pos_i+1):end,vel_i,a_i)]
            elseif vel_size/(vel_max-vel_min) == split_dim
                q1 = 0;
                q2 = 0;
                vel_segnum = vel_segnum+1;
                vel_nodes = [vel_nodes(1:vel_i),vel_nodes(vel_i)+vel_size/2,vel_nodes((vel_i+1):end)];
                for kk = 1:j
                    if sample(kk,4)==pos_i
                        if sample(kk,6)==a_i
                            if sample(kk,2)>vel_nodes(vel_i)&&sample(kk,2)<=vel_nodes(vel_i+1)
                                q1 = [q1,sample(kk,7)];
                            elseif sample(kk,2)> vel_nodes(vel_i+1) && sample(kk,2)<= vel_nodes(vel_i+2)
                                q2 = [q2,sample(kk,7)];
                            end
                        end
                    end
                end
                Q(pos_i,:,a_i) = [Q(pos_i,1:(vel_i-1),a_i);mean(q1);mean(q2);Q(pos_i,(vel_i+1):end,a_i)];
                S(pos_i,:,a_i) = [S(pos_i,1:(vel_i-1),a_i);std(q1);std(q2);S(pos_i,(vel_i+1):end,a_i)];
                n(pos_i,:,a_i) = [n(pos_i,1:(vel_i-1),a_i);length(q1);length(q2);n(pos_i,(vel_i+1):end,a_i)];
            elseif a_size/(a_max-a_min) == split_dim
                q1 = 0;
                q2 = 0;
                a_segnum = a_segnum+1;
                a_nodes = [a_nodes(1:a_i),a_nodes(a_i)+a_size/2,a_nodes((a_i+1):end)];
                for kk = 1:j
                    if sample(kk,4)==pos_i
                        if sample(kk,5)==vel_i
                            if sample(kk,3)>a_nodes(a_i)&&sample(kk,3)<=a_nodes(a_i+1)
                                q1 = [q1,sample(kk,7)];
                            elseif sample(kk,3)> a_nodes(a_i+1) && sample(kk,3)<= a_nodes(a_i+2)
                                q2 = [q2,sample(kk,7)];
                            end
                        end
                    end
                end
                Q(pos_i,vel_i,:)=[Q(pos_i,vel_i,1:(a_i-1));mean(q1);mean(q2);Q(pos_i,vel_i,(a_i+1):end)];
                S(pos_i,vel_i,:)=[S(pos_i,vel_i,1:(a_i-1));std(q1);std(q2);S(pos_i,vel_i,(a_i+1):end)];
                n(pos_i,vel_i,:)=[n(pos_i,vel_i,1:(a_i-1));length(q1);length(q2);n(pos_i,vel_i,(a_i+1):end)];
            end
        end
        
        % s=s'
        pos = pos_new;
        vel = vel_new;  
        pos_ten(i,j) = pos;
        vel_ten(i,j) = vel;
        r_ten(i,j) = r;
        %if pos == 0
        %    break
        %end-
    end
    
    % test 
  %  for j = 1:test_num
  %      R_test[i,j] =; 
end

%% plot acc_reward