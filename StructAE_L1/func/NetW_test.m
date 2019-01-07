function [accuracy, nmi, ARI, Precision, Recall, Fscore] = NetW_test(data, net, opts, labels)
% ----------------testing NN
% input: data:  each column is a data point
%        net:   the parameters of neural network
%        opts:  the parameters of algorithms
% written by Xi Peng
% Dec. 2015, I2R, A*STAR
% ----------------
if ~isfield(opts,'nCluster')
    opts.nCluster=0;
end

for k = 1:size(net.W,1)
    Xa = data;
    % --- calculate the final output of the network
    for j = 2:net.nEnclayer+1
        %         Xa = net.W{k,j}*Xa + repmat(net.b{k,j},1,size(Xa,2));
        [~, Xa] = feedforward(Xa, net.W{k,j}, net.b{k,j}, net, net.actfun{j-1});
    end
    nPos = 1;
    if opts.nCluster == 0 || opts.nCluster == 1
        predict_label = kmeans(Xa',length(unique(labels)),'start','sample','maxiter',2000,'replicates',200,'EmptyAction','singleton');
        predict_label = reshape(predict_label,size(labels,1),[]);
        [accuracy(nPos,k), nmi(nPos,k), ARI(nPos,k), Precision(nPos,k), Recall(nPos,k), Fscore(nPos,k)]= CalMetricOfCluster(predict_label,labels);
        fprintf('\n * kmeans: acc = %f, nmi = %f, ARI = %f, Precision = %f, Recall = %f, Fscore = %f \n',accuracy(nPos,k), nmi(nPos,k), ARI(nPos,k), Precision(nPos,k), Recall(nPos,k),Fscore(nPos,k));
        nPos = nPos + 1;
    end
    if opts.nCluster == 0 || opts.nCluster == 2
        for i = 1:size(Xa,2)
            if i ==1
                tmp = sparsecode_func(Xa(:,i), Xa(:, i+1:end), opts.gamma, opts.tolerance);
                coef(:,i) = [0; tmp];
            else
                tmp = sparsecode_func(Xa(:,i), [Xa(:, 1:i-1) Xa(:, i+1:end)], opts.gamma, opts.tolerance);
                coef(:,i) = [tmp(1:i-1); 0; tmp(i:end)];
            end
        end
        CKSym = BuildAdjacency(coef,0);
        predict_label = SC(CKSym,length(unique(labels)));
        predict_label = reshape(predict_label,1,[]);
        
        [accuracy(nPos,k), nmi(nPos,k), ARI(nPos,k), Precision(nPos,k), Recall(nPos,k), Fscore(nPos,k)]= CalMetricOfCluster(predict_label,labels);
        fprintf(' * SC: acc = %f, nmi = %f, ARI = %f, Precision = %f, Recall = %f, Fscore = %f \n',accuracy(nPos,k), nmi(nPos,k), ARI(nPos,k), Precision(nPos,k), Recall(nPos,k),Fscore(nPos,k));
    end
end