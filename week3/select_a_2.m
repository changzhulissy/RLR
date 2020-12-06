function a = select_a_2(Q,S,n,a_nodes,a_segnum,a_min,a_max)


Q_rand = zeros(a_segnum,1);

%a_vec = linspace(a_min,a_max,a_segnum);
for i = 1:a_segnum
     freedeg = n(i);
     if freedeg == 0||freedeg==1
         freedeg = 2;
     end
     t = trnd(freedeg-1,1);
     Q_rand(i) = Q(i)-t*sqrt(S(i)^2/freedeg);  
     
end
[~,a_idx] = max(Q_rand);
a = a_nodes(a_idx);

end