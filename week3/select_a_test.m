function a = select_a_2(Q,S,n,a_nodes,a_segnum,a_min,a_max)





[~,a_idx] = max(Q);
a = a_nodes(a_idx);

end