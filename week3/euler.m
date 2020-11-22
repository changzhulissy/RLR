function [new_pos, new_vel] = euler(pos,vel,acc)
delta_t = 0.001;
new_vel = vel+delta_t*acc;
new_pos = pos + delta_t*vel+0.5*delta_t*delta_t*acc;
if new_pos > pi
   new_pos = mod(new_pos,2*pi);
elseif new_pos < -pi
   new_pos = mod(new_pos,-2*pi);
end
end