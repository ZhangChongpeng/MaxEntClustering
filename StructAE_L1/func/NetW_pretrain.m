function net = NetW_pretrain(X, net, opts)
% ----------------pretraining NN
% input: X: each column is a data point
%        net:   the parameters of neural network
%        opts:  the parameters of algorithms
%        mnet:  the parameters of autoencoder
% written by Xi Peng
% Dec. 2015, I2R, A*STAR

data = X;
for i = 1:net.nEnclayer% nEnclayer denotes the no of autoencoders
    fprintf('\n  | pretraining the %d-th AE\n', i);
    mnet.neuron    = [net.neuron(i) net.neuron(i+1) net.neuron(net.nlayer-i+1)];
    mnet.knn=net.knn;
    mnet.nlayer    = length(mnet.neuron);
    mnet.nEnclayer = (mnet.nlayer+1)/2;
    mnet.obj_engy  = zeros(max(opts.iter),1);
    mnet.actfun{1}  = net.actfun{i};
    mnet.actfun{2}  = net.actfun{net.nlayer-i};
    for j = 2:mnet.nlayer
        [mnet.W{1,j}, mnet.b{1,j}]= InitNet(mnet.neuron(j), mnet.neuron(j-1), opts.initmod);
    end
    clear j;
    [mnet, opts] = NetW_train(data, mnet, opts);
    [~, data] = feedforward(data, mnet.W{end,2}, mnet.b{end,2}, mnet, mnet.actfun{1});% encode
    net.W{i+1} = mnet.W{end,2};
    net.W{net.nlayer-i+1} = mnet.W{end,3};
    net.b{i+1} = mnet.b{end,2};
    net.b{net.nlayer-i+1} = mnet.b{end,3};  
    net.AEobj_engy{i} = mnet.obj_engy;
    clear mnet;
end
