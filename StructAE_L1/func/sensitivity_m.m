function delta = sensitivity_m(W, Delta_bp, z, net, varargin)
% ---------------- calculate the sensitivity of layer m
% input: W: W at layer m+1
%        delta_bp: sensitivity at m+1 layer
%        z: net input at m layer, i.e., z = w^{m}h^{m-1}+b^{m}
% output: Delta: the sensitivity at m layer
% written by Xi Peng
% Dec. 2015, I2R, A*STAR
% ---------------- 
if length(varargin)==0 % default activation func
    sd = 1 - tanh(z).^2;  % gradient of s(z_{m})
else
    switch cell2mat(varargin{1})
        case 'tanh'
            sd = 1 - tanh(z).^2;
        case 'sigmoid' 
            tmp = 1 ./ (1 + exp(-z));
            sd = tmp .* (1-tmp);
        case 'nssigmoid'
            [~, sd] = nonsaturate_sigmoid_act(z);
        case 'relu'            
            alpha = net.alpha;
            sd = 1./(1+exp(-alpha*z));
%         otherwise
%             fprintf('The specified activation fun (%s) is not support!\n', varargin);
    end
end
delta = W' * Delta_bp .* sd;   


