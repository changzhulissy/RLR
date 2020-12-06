function acc = dynamics(vel,pos,torque)
mu = 0.01;
m = 1;
g = 9.8;
l = 1;
acc = (-mu*vel+m*g*l*sin(pos)+torque)/(m*l^2);
end