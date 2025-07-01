%This function implements Convolution, Batch Normalization, and ReLU activation operations.
function [dlY,state] = perceptron(dlX,parameters,state,isTraining)
% Convolution.
W = parameters.Conv.Weights;
B = parameters.Conv.Bias;
 

 
dlY = dlconv2(dlX,W,B);%Call rewritten conv2.
 
% Batch normalization. Update batch normalization state when training.
offset          = parameters.BatchNorm.Offset;
scale           = parameters.BatchNorm.Scale;
trainedMean     = state.BatchNorm.TrainedMean;
trainedVariance = state.BatchNorm.TrainedVariance;
if isTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
    
    % Update state.
    state.BatchNorm.TrainedMean = trainedMean;
    state.BatchNorm.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
end

% ReLU.
dlY = relu(dlY);
end