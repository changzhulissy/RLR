pos = pi;
vel = 0;
sample_test= zeros(iter_num,6);   
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
        torque = select_a_test(Q(pos_i,vel_i,:),S(pos_i,vel_i,:),n(pos_i,vel_i,:),a_nodes,a_segnum,a_min,a_max);
        a_i = (torque-a_min)/0.5+1;

        sample_test(j,:) = [pos,vel,torque,pos_i,vel_i,a_i];
        [pos_new,vel_new,~] = euler_pend(pos,vel,torque,0.1);
        pos = pos_new;
        vel = vel_new;
end


%% plot acc_reward
% Set up video
v=VideoWriter('pendulum_test.avi');
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
    plot([0 l*sin(sample_test(ii,1))],[0 l*cos(sample_test(ii,1))],'b','LineWidth',3); % plots rod
    plot(l*sin(sample_test(ii,1)),l*cos(sample_test(ii,1)),'Marker','o','MarkerSize',20,'MarkerFaceColor','r','MarkerEdgeColor','r'); % plots bob
    text(0,0,"Timer: "+num2str(0.1*ii)+" s");
    %plot(0,0,'Marker','0','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k'); % plots pin
    xlim([-l-0.2*l l+0.2*l]);
    ylim([-l-0.2*l l+0.2*l]);
    title('Simple Pendulum');
    frame=getframe(gcf);
    writeVideo(v,frame);
end

close(v);