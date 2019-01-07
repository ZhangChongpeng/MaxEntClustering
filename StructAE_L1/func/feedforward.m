function [z, h, net] = feedforward(x, W, b, net, varargin)
% ----------------training NN
% input: (x):  the output of the previous layer (layer)
%        net:   the parameter of neural network
% output: (h): the output at layer m
%         (z): the net input to network at layer m
% written by Xi Peng
% Sep. 2015, I2R, A*STAR
% ----------------

N = size(x,2);
z = W*x + repmat(b,1,N);
if length(varargin) == 0
    h = tanh(z); % default 
else
    switch varargin{1}
        case 'tanh'
            h = tanh(z); 
        case 'sigmoid'           
            h = 1./(1+exp(-z));
        case 'nssigmoid'
            [h] = nonsaturate_sigmoid_act(z);
        case 'relu'
            if isfield(net, 'alpha')
                alpha = net.alpha;
            else
                alpha = 10;% default
                net.alpha = alpha;
            end
            h = log(1+exp(alpha*z))/alpha;
        case 'linear'
            h = z;
        otherwise
            fprintf('The specified activation fun (%s) is not support!\n', varargin);
    end
end

