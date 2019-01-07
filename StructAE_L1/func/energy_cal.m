function J = energy_cal(net,opts, H, iter_ind)
% ----------------
% input: net:  the network config
%        opts: the parameter
%        H: the collection of output neural netw
%        iter_ind: the index of iter
% output: J - total energy, scalar 
% written by Xi Peng
% Dec. 2015, I2R, A*STAR
% ----------------
lambda2 = 1e-3;
if isfield(opts,'lambda2')
    lambda2= opts.lambda2;
end

J1 = 0.5 * norm(H{1} - H{net.nlayer}, 'fro')^2;
J2 = 0;
J3 = 0;

S=net.S;
D=diag(sum(S,2));
L=D-S;

for i =net.nEnclayer
    J2=J2+trace(H{i}*L*H{i}');
    J3 = J3 + norm(net.W{iter_ind,i},'fro')^2 + norm(net.b{iter_ind,i})^2;
end;
J2 = 0.5*opts.lambda1*J2;
J3 = 0.5*lambda2*J3;

J = J1+J2+J3;
