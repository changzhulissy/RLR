function [new_pos, new_vel, pos_mat] = euler_pend(pos,vel,torque,t)
delta_t = 0.001;
pos_mat = zeros(t/delta_t,1);
for i =1:t/delta_t
    
    acc = dynamics(vel,pos,torque);
    
    % vel [-2*pi,2*pi]
    new_vel = vel+delta_t*acc;
    if new_vel > 2*pi
           new_vel = 2*pi;
    elseif new_vel < -2*pi
           new_vel = -2*pi;
    end
    % pos [-pi,pi]
    new_pos = pos + delta_t*vel+0.5*delta_t*delta_t*acc;
    if new_pos > pi
        new_pos = new_pos-2*pi;
    elseif new_pos < -pi
        new_pos = new_pos+2*pi;
    end
    
    vel = new_vel;
    pos = new_pos;
    pos_mat(i) = pos;
end
end