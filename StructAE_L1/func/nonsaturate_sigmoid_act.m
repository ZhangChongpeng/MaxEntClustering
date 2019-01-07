function [a, grad] = nonsaturate_sigmoid_act(X)

[m, n] = size(X);
a = zeros(m, n);
for i = 1:m
    for j = 1:n
        t_x = X(i, j);
        if t_x >= 0
            a(i, j) = do_newton(t_x);
        else
            a(i, j) = - do_newton(-t_x);
        end
    end
end

if nargout > 1
    grad = 1 ./ (a.^2 + 1);
end

end

function y = do_newton(x)

y = x;
g_y = y*y*y/3 + y;
while abs(g_y - x) > 1e-6;
    y = (2*y*y*y/3 + x)/(y*y + 1);
    g_y = y*y*y/3 + y;
end

end