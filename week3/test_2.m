clear; clc;
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
%select_seg_a = 20;
alpha = 0.1;%forget_rate
gamma = 0.85;%discount_rate
thr_error = 5.3;

%sample(j,:) = [pos,vel,torque,pos_i,vel_i,a_i,q]

%% Q-Learning with variable resolution function
%dbstop if naninf
pos_segnum = 2; % initial segmentation of position (state1)
pos_nodes = [pos_min,(pos_min+pos_max)/2,pos_max];
vel_segnum = 2; % initial segmentation of velocity (state2)
vel_nodes = [vel_min,(vel_min+vel_max)/2,vel_max];

a_segnum = 21; % initial segmentation of torque (action)
a_nodes = linspace(a_min,a_max,a_segnum);

Q = zeros(pos_segnum,vel_segnum,a_segnum); % initialization Q_mean statistic
S = ones(pos_segnum,vel_segnum,a_segnum); % initialization S variance statisctic
n = zeros(pos_segnum,vel_segnum,a_segnum); % initial sample number (n)

pos_ten = zeros(episode_num, iter_num);
vel_ten = zeros(episode_num,iter_num);
r_ten = zeros(episode_num,iter_num);
%%
for i = 1:episode_num
    %print(num2str(i));
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
        torque = select_a_2(Q(pos_i,vel_i,:),S(pos_i,vel_i,:),n(pos_i,vel_i,:),a_nodes,a_segnum,a_min,a_max);
        a_i = (torque-a_min)*(a_segnum-1)/(a_max-a_min)+1;
        [pos_new,vel_new,~] = euler_pend(pos,vel,torque,0.1);
        r = reward(pos_new,vel_new);
        
        % exploration policy: estimate maximum Qmax 
        for kk = 1:pos_segnum
            if pos_new > pos_nodes(kk)
                if pos_new <= pos_nodes(kk+1)
                   pos_new_i =kk;
                end
            end
        end
        for mm = 1:vel_segnum
            if vel_new > vel_nodes(mm)
                if vel_new <= vel_nodes(mm+1)
                    vel_new_i = mm;
                end
            end
        end     
        Q_perm = zeros(a_segnum,1);
        for jj = 1:a_segnum
            Q_perm(jj) = Q(pos_new_i,vel_new_i,jj);
        end
        q = r+gamma*max(Q_perm);

        n(pos_i,vel_i,a_i) = n(pos_i,vel_i,a_i)+1;
        
        sample(j,:) = [pos,vel,torque,pos_i,vel_i,a_i,q]; % update only in one episode
        S_i = S(pos_i,vel_i,a_i);
        Q_i = Q(pos_i,vel_i,a_i); 
        if n(pos_i,vel_i,a_i) < 21
            forget_rate = 0.3;
        else            
            forget_rate = 1/(n(pos_i,vel_i,a_i)-10);
        end
        Q(pos_i,vel_i,a_i) = Q_i+forget_rate*(q-Q_i);
        S(pos_i,vel_i,a_i) = sqrt(S_i^2+forget_rate*((q-Q_i)^2-S_i^2));
        %Q(pos_i,vel_i,a_i) = Q_i+alpha*(q-Q_i);
        %S(pos_i,vel_i,a_i) = sqrt(S_i^2+alpha*((q-Q_i)^2-S_i^2));
        %variable resolution
        if (q-Q(pos_i,vel_i,a_i))^2>=thr_error && n(pos_i,vel_i,a_i)>20 
            pos_size = pos_nodes(pos_i+1)-pos_nodes(pos_i);
            vel_size = vel_nodes(vel_i+1)-vel_nodes(vel_i);         
            split_dim = max([pos_size/(2*pi),vel_size/(vel_max-vel_min)]);
           
            if pos_size/(2*pi) == split_dim %&& size(Q,1)<21
                pos_segnum = pos_segnum+1;
                pos_nodes = [pos_nodes(1:pos_i),pos_nodes(pos_i)+pos_size/2,pos_nodes((pos_i+1):end)];
                q1 = [];
                q2 = [];               
                for kk = 1:j
                    if sample(kk,5)==vel_i
                        if sample(kk,6)==a_i
                            if sample(kk,1)>pos_nodes(pos_i) && sample(kk,1)<=pos_nodes(pos_i+1)
                                q1 = [q1,sample(kk,7)];
                            elseif sample(kk,1)> pos_nodes(pos_i+1) && sample(kk,1)<= pos_nodes(pos_i+2)
                                q2 = [q2,sample(kk,7)];
                            end
                        end
                    end
                end

                Q_new = zeros(pos_segnum, vel_segnum, a_segnum);
                S_new = zeros(pos_segnum, vel_segnum, a_segnum);
                n_new = zeros(pos_segnum, vel_segnum, a_segnum);
                for ii = 1:vel_segnum
                    for jj = 1:a_segnum
                        Q_new(:,ii,jj) = [squeeze(Q(1:(pos_i-1),ii,jj));Q(pos_i,ii,jj);Q(pos_i,ii,jj);squeeze(Q((pos_i+1):end,ii,jj))];
                        S_new(:,ii,jj) = [squeeze(S(1:(pos_i-1),ii,jj));S(pos_i,ii,jj);S(pos_i,ii,jj);squeeze(S((pos_i+1):end,ii,jj))];
                        n_new(:,ii,jj) = [squeeze(n(1:(pos_i-1),ii,jj));n(pos_i,ii,jj);n(pos_i,ii,jj);squeeze(n((pos_i+1):end,ii,jj))];
                    end
                end
                Q_new(:,vel_i,a_i) = [squeeze(Q(1:(pos_i-1),vel_i,a_i));mean(q1);mean(q2);squeeze(Q((pos_i+1):end,vel_i,a_i))];              
                S_new(:,vel_i,a_i) = [squeeze(S(1:(pos_i-1),vel_i,a_i));std(q1);std(q2);squeeze(S((pos_i+1):end,vel_i,a_i))];
                n_new(:,vel_i,a_i) = [squeeze(n(1:(pos_i-1),vel_i,a_i));length(q1);length(q2);squeeze(n((pos_i+1):end,vel_i,a_i))];
                Q = Q_new;
                S = S_new;
                n = n_new;
            elseif vel_size/(vel_max-vel_min) == split_dim %&& size(Q,2)<21
                q1 = [];
                q2 = [];
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
                Q_new = zeros(pos_segnum, vel_segnum, a_segnum);
                S_new = zeros(pos_segnum, vel_segnum, a_segnum);
                n_new = zeros(pos_segnum, vel_segnum, a_segnum);
                for ii = 1:pos_segnum
                    for jj = 1:a_segnum
                        Q_new(ii,:,jj) = [squeeze(Q(ii,1:(vel_i-1),jj))';Q(ii,vel_i,jj);Q(ii,vel_i,jj);squeeze(Q(ii,(vel_i+1):end,jj))'];
                        S_new(ii,:,jj) = [squeeze(S(ii,1:(vel_i-1),jj))';S(ii,vel_i,jj);S(ii,vel_i,jj);squeeze(S(ii,(vel_i+1):end,jj))'];
                        n_new(ii,:,jj) = [squeeze(n(ii,1:(vel_i-1),jj))';n(ii,vel_i,jj);n(ii,vel_i,jj);squeeze(n(ii,(vel_i+1):end,jj))'];
                    end
                end
                Q_new(pos_i,:,a_i) = [squeeze(Q(pos_i,1:(vel_i-1),a_i))';mean(q1);mean(q2);squeeze(Q(pos_i,(vel_i+1):end,a_i))'];              
                S_new(pos_i,:,a_i) = [squeeze(S(pos_i,1:(vel_i-1),a_i))';std(q1);std(q2);squeeze(S(pos_i,(vel_i+1):end,a_i))'];
                n_new(pos_i,:,a_i) = [squeeze(n(pos_i,1:(vel_i-1),a_i))';length(q1);length(q2);squeeze(n(pos_i,(vel_i+1):end,a_i))'];
                Q = Q_new;
                S = S_new;
                n = n_new;                  
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
% Set up video
v=VideoWriter('pendulum.avi');
v.FrameRate=10;
open(v);
mu = 0.01;
m = 1;
g = 9.8;
l = 1;
for ii=1:iter_num
    % Iteratively solve equations of motion using Euler's Method
   % theta(n+1,:)=theta(n,:)+omega(n,:)*tStep; % new angular position
   % omega(n+1,:)=omega(n,:)+alpha(n,:)*tStep; % new angular velocity
   % alpha(n+1,:)=(-g*sin(theta(n+1,:)))/l-c*omega(n+1,:); % new angular acceleration
    %sample(j,:) = [pos,vel,torque,pos_i,vel_i,a_i,q]; 
    % Plot everything for the video
    hold on;
    fill([-l-0.2*l l+0.2*l l+0.2*l -l-0.2*l],[-l-0.2*l -l-0.2*l l+0.2*l l+0.2*l],'w'); % clears background
    plot([0 l*sin(sample(ii,1))],[0 l*cos(sample(ii,1))],'b','LineWidth',3); % plots rod
    plot(l*sin(sample(ii,1)),l*cos(sample(ii,1)),'Marker','o','MarkerSize',20,'MarkerFaceColor','r','MarkerEdgeColor','r'); % plots bob
    text(0,0,"Timer: "+num2str(0.1*ii)+" s");
    %plot(0,0,'Marker','0','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k'); % plots pin
    xlim([-l-0.2*l l+0.2*l]);
    ylim([-l-0.2*l l+0.2*l]);
    title('Simple Pendulum');
    frame=getframe(gcf);
    writeVideo(v,frame);
end

close(v);
%% plot r
r_acc = sum(r_ten,2);
plot(r_acc)