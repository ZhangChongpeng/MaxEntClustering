function flag = checkNetWconfig_func(net)
% check the network configuration 
flag = true;
if length(net.actfun) ~= net.nlayer - 1
    fprintf('Only %d activation functions are required!\n', net.nlayer - 1);
    flag = false;
end
for i = 1:length(net.actfun)
    switch net.actfun{i}
        case 'tanh'
        case 'sigmoid'
        case 'nssigmoid'
        case 'relu'
        otherwise
            fprintf('Only sigmoid, nonsaturate_sigmoid, tanh, and relu functiosn are supported\n');
            flag = false;
    end
end