function coef = sparsecode_func(x, D, lambda, tolerance)

options.maxIteration = 600;
options.isNonnegative = false;
STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
options.stoppingCriterion = STOPPING_OBJECTIVE_VALUE;

[coef, tmp_iter] = SolveHomotopy(D, x, ...
                    'maxIteration', options.maxIteration,...
                    'isNonnegative', options.isNonnegative, ...
                    'lambda', lambda, ...
                    'tolerance', tolerance);
                
if sum(coef==0)==length(coef)
    fprintf('--- error in sparse code (sparsecode_func.m), where the sparse code of one sample is with all zeroes !!!\n');
    fprintf('--- randomly set all elements as eps!\n');
    coef = eps*ones(length(coef),1);    
end
