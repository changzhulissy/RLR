function a = select_a(Q,S,n,select_seg_a,a_nodes,a_segnum,a_min,a_max)


Q_rand = zeros(select_seg_a,1);


for i = 1:select_seg_a   
    a = a_min+(a_max-a_min)*i/select_seg_a;
    for j = 1:a_segnum
        if a>a_nodes(j)
            if a<=a_nodes(j+1)
                freedeg = n(j);
                t = trnd(freedeg);
                Q_rand(i) = Q(j)-t*sqrt(S(j)^2/freedeg);  
            end
        end
    end    
end
a = max(Q_rand);
end