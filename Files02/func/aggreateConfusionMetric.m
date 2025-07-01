function cmat = aggreateConfusionMetric(cmat,YTest,YPred)
YTest = gather(extractdata(YTest));
YPred = gather(extractdata(YPred));
[m,n] = size(cmat);%%% Aggregate confusion matrices.
cmat = cmat + full(sparse(YTest,YPred,1,m,n));
end