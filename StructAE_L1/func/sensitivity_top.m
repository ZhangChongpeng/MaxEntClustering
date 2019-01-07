function delta = sensitivity_top(z_M, h_1, h_M, net, varargin)
% ---------------- calculate the sensitivity of top layer 
% input: 
%        z_M:  net input at M layer, i.e., z_M = w_{M}h_{M-1}+b_{M}
%        h_1:  input
%        h_M:  final representation of x, i.e., output of the top layer
% output: Delta: the sensitivity at M layer (i.e., the top layer)
% written by Xi Peng
% Dec. 2015, I2R, A*STAR
% ---------------- 

gd = h_M - h_1;

if  length(varargin)==0 % default 
    sd = 1 - tanh(z_M).^2;  % gradient of s(z_{m}), default func is tanh
else
    switch cell2mat(varargin{1})
        case 'tanh'
            sd = 1 - tanh(z_M).^2;
        case 'sigmoid' 
            tmp = 1 ./ (1 + exp(-z_M));
            sd = tmp .* (1-tmp);
        case 'nssigmoid'
            [~, sd] = nonsaturate_sigmoid_act(z_M);
        case 'relu'            
            alpha = net.alpha;
            sd = 1./(1+exp(-alpha*z_M));
        otherwise
            fprintf('The specified activation fun (%s) is not support!\n', varargin);
    end
end
delta = gd .* sd;      % sensitivity






