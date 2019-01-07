clear;clc;
addpath('./func/');
addpath('../usages/');
addpath('../data/');
% --- the directory to store the data in .mat, where each column is a data
% point
opts.name     = 'YaleB_LPQ_PCA300_TValid';
opts.dim      = 300;
load (opts.name);
% nor=zscore(X);  % 数据标准化
% data =nor;
% labels = Y;
% [pc,score,latent,tsquare]=pca(data);
% z=cumsum(latent)./sum(latent);
% feature_after_PCA=score(:,1:350);
% data=feature_after_PCA';
data=tr_dat;
labels=tr_labels;
% ---------------- parameters setup
% --- the parameters of neural netw
opts.initmod   = 1;       % init method 0: zeros; 1: eye; 2: [-x x]. 3: [-0.5  0.5]
% --- parameter for BP
par.mu         = [0.001 0.01]; % learning rate, can be smaller锛?default 2^(10)
% par.mu         = [0.01 0.001 0.0001 2^(-10) 2^(-11) 2^(-12)]; % learning rate, can be smaller锛?default 2^(10)
opts.iter      = [5];     % maximal iterative number.
opts.updateNo  = [100];   % store (W,b) after updateNo updates
opts.error     = 1e-3;    % convergence error, if |J(t+1)-J(t)|<opts.error.
% --- parameters of our objective function
par.lambda1   = [0.001 0.01 0.1 0.05 0.005 0.0005];% regularized on \|H-HC\|_{F}
par.lambda2   = [0.001 0.01 0.1 0.05 0.005 0.0005];% regularized on \|H-HC\|_{F}
% --- the parameters of L1 solver, Homotopy is used
opts.gamma     = [0.5];    % sparsity parameter of L1 这两个参数对结果有影响
opts.tolerance = [0.1]; % tolerance of L1

net.knn=10;  % parameter k of similarity matrix S 

% --- neural network
net.neuron         = [300 200 150 200 300]; % number of neurons at layer 1, 2, and 3, i.e., input
net.nlayer         = length(net.neuron);    % number of layers, the first layer is the input
net.nEnclayer      = (net.nlayer-1)/2;      % number of encoder layers
net.obj_engy       = zeros(max(opts.iter),1);
% --- the activation functions used at different layers
% net.actfun         = {'nssigmoid', 'nssigmoid', 'nssigmoid', 'nssigmoid'}; % sigmoid, tanh, relu, and nssigmoid, default tanh
net.actfun         = {'tanh', 'tanh','tanh', 'tanh'}; % sigmoid, tanh, relu, and nssigmoid, default tanh
net.alpha          = 10; % the shapeness of relu; default 10
% ---------------- check the configuration of network
if ~checkNetWconfig_func(net)
    return;
end
% ---------------- data process if the dim of input is larger than neuron(1)
if net.neuron(1) < size(data,1)
    data = data(1:net.neuron(1),:);
    data = bsxfun(@minus, data, mean(data,2));
elseif net.neuron(1) > size(data,1)
    fprintf('The number of neuron at the first layer should be smaller than the dimension of inputs\n');
    return;
end

% ---------------- learning with our neural network
for j=1:length(par.mu)
    opts.mu = par.mu(j);
for i = 1:length(par.lambda1) % learning rate
    opts.lambda1 = par.lambda1(i);
    for ii = 1:length(par.lambda2)
        opts.lambda2 = par.lambda2(ii);
        fprintf(' * learning rate=%1.2e, lambda1=%1.2e, lambda2=%1.2e \n', opts.mu,opts.lambda1,opts.lambda2);
        % --- pretrain the neural network
        if isfield(net, 'W')
            net = rmfield(net,'W');
            net = rmfield(net,'b');
        end
        [net] = NetW_pretrain(data, net, opts);
        % --- fine tuning the network
        fprintf(' * pretraining completed, begin to fine tune!\n');
        [net, opts] = NetW_train(data, net, opts);
        % --- testing
        [accuracy, nmi, ARI, Precision, Recall, Fscore] = NetW_test(data, net, opts, labels);
        save(['./Result_TunP_Source/PARTY_' opts.name '_#layer' num2str(length(net.nlayer)) '_mu' num2str(opts.mu, '%1.2e') '_lbd1' num2str(opts.lambda1, '%1.2e') '_lbd2' num2str(opts.lambda2, '%1.2e')  '_#iter_TunP' num2str(max(opts.iter)) '.mat'], 'net', 'opts', 'accuracy', 'nmi', 'ARI', 'Precision', 'Recall', 'Fscore'); 
    end
end
end;