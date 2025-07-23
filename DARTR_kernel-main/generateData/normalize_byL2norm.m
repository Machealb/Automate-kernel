function [u_normalized, f_normalized] = normalize_byL2norm(u, f, dx)
d = numel(size(u));
n = vecnorm(u, 2, d)*sqrt(dx);
u_normalized = u./n;  
f_normalized = f./n;

end