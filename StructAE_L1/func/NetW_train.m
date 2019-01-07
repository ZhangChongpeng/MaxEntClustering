function [net, opts] = NetW_train(X, net, opts)
% ----------------training NN
% input: X: each column is a data point
%        net:   the parameters of neural network
%        opts:  the parameters of algorithms
% written by Xi Peng
% Dec. 2015, I2R, A*STAR

% update the neural network sample by sample
% ----------------

% --- initializing the network
N = size(X,2);

dW    = cell(1,net.nlayer); % gradient w.r.t. W
db    = cell(1,net.nlayer); % gradient w.r.t. b
delta = cell(1,net.nlayer); % sensitivity of network

% --- initializing F(x)
Z    = cell(1,net.nlayer);  % the net input to network, i.e., z = Wx+b
H    = cell(1,net.nlayer);  % the output of network, i.e., h = s(z)
coef = zeros(N);            % structure prior.
H{1} = X;
clear X;
% --- initializing H(x)
for i = 2:net.nlayer
    dW{i} = zeros(size(net.W{1,i}));
    db{i} = zeros(size(net.b{1,i}));
    [Z{i}, H{i}, net] = feedforward(H{i-1}, net.W{1,i}, net.b{1,i}, net, net.actfun{i-1});%
end

% --- initializing Compute the sparsity S
net.S=constructEntropy(H{net.nEnclayer},net.knn);

% --- training the network
iter_ind = 1; % record the (W,b) at the ii-th iteration
% m_begin = 1;
% m_end = opts.updateNo(iter_ind);

pos = 1;
for i =1:opts.iter
    m_order = randperm(N);
    for j = 1:N
        % --- randomly choosing a data point
        idx = m_order(j);
        % --- feedforward
        if j > 1
            for k = 2:net.nlayer
                [Z{k}(:,idx), H{k}(:,idx)] = feedforward(H{k-1}(:,idx), net.W{iter_ind,k}, net.b{iter_ind,k}, net, net.actfun{k-1});
            end
        end
        % --- back propagation
        for k = net.nlayer:-1:2
            % --- sensitivity calculation
            if k == net.nlayer % top layer
                delta{k} = sensitivity_top(Z{k}(:,idx), H{1}(:,idx), H{k}(:,idx), net, net.actfun(k-1));
            else               % other layers
                delta{k} = sensitivity_m(net.W{iter_ind,k+1}, delta{k+1}, Z{k}(:,idx), net, net.actfun(k-1));
            end
            if k <= net.nEnclayer
                if k == net.nEnclayer
                    [Lambda{k}] = sensitivity_embed_top(Z{k}(:,idx), idx, H{k}, net, opts, net.actfun(k-1));
                else
                    [Lambda{k}] = sensitivity_embed_m(net.W{iter_ind,k+1}, Lambda{k+1}, Z{k}(:,idx), net, net.actfun(k-1));
                end
                % --- gradient calculation
                db{k} = delta{k} + opts.lambda1*Lambda{k};
                dW{k} = db{k}*H{k-1}(:,idx)'+opts.lambda2 * net.W{iter_ind,k};   
                db{k}=db{k} + opts.lambda2 * net.b{iter_ind,k};
            else
                db{k} = delta{k};
                dW{k} = db{k}*H{k-1}(:,idx)'+opts.lambda2 * net.W{iter_ind,k};
                db{k}=db{k} + opts.lambda2 * net.b{iter_ind,k};
            end
            % --- update network
            net.W{iter_ind,k} = net.W{iter_ind,k} - opts.mu * dW{k};
            net.b{iter_ind,k} = net.b{iter_ind,k} - opts.mu * db{k};
        end
        for k = 2:net.nlayer
            [Z{k}, H{k}] = feedforward(H{k-1}, net.W{iter_ind,k}, net.b{iter_ind,k}, net, net.actfun{k-1});%
        end
        net.obj_engy(pos) = energy_cal(net, opts, H, iter_ind);
        if ~mod(pos,100)
            fprintf(' | the %5d-th update: | Energy is about %4.2f\n', pos, net.obj_engy(pos));
        end
        if i > 1 && pos > 1 && ((abs(net.obj_engy(pos) - net.obj_engy(pos-1)) < opts.error))
            fprintf('convergent at the %d-th update (%d epoch)\n', pos, i);
            return;
        end
        if ~mod(pos,opts.updateNo) && opts.iter > 1 && iter_ind<opts.iter*opts.updateNo
            for k = 1:net.nlayer
                net.W{iter_ind+1,k} = net.W{iter_ind,k};
                net.b{iter_ind+1,k} = net.b{iter_ind,k};
            end
            iter_ind = iter_ind + 1;
            fprintf('                  *  the %2d-th (W, b) is stored!\n', iter_ind-1);
        end
        pos = pos + 1;
    end
    
    if i<opts.iter
        for k = 2:net.nlayer
            [Z{k}(:,idx), H{k}(:,idx)] = feedforward(H{k-1}(:,idx), net.W{iter_ind,k}, net.b{iter_ind,k}, net, net.actfun{k-1});
        end     
        % --- update the sparsity S
        net.S=constructEntropy(H{net.nEnclayer},net.knn);
    end
    
end
