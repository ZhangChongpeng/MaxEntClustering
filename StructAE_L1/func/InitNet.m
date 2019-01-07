function [Wm, Bm] = InitNet(n_current, n_previous, mark)
% ----------------
% input: n_current:  the number of neurons at the current layer
%        n_previous: the number of neurons at the previous layer
% output: Wm: follows the uniform distribution between [b_min, b_max]
%         Bm: bais
% written by Xi Peng
% June. 2015, I2R, A*STAR
% ----------------
rand('state',6);% to reproduce our result
switch mark
    case 0 % 0: zeros
        Wm = zeros(n_current, n_previous);
        Bm = zeros(n_current, 1);
    case 1 % 1: eyes
        Wm = eye(n_current, n_previous);
        Bm = zeros(n_current, 1);
    case 2 % 
        r = sqrt(6)/(sqrt(n_current + n_previous));
        Wm = -r + 2*r*rand(n_current, n_previous);
        Bm = zeros(n_current,1);
    case 3 % 3: [-0.5  0.5]
        Wm = 0.5 - rand(n_current, n_previous);
        Bm = zeros(n_current, 1);
end

