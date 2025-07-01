function [dlY,state] = sharedMLP(dlX,parameters,state,isTraining)
dlY = dlX;
%Shared Multi-Layer Perceptron (MLP) function. %%Specifically, it builds the forward propagation process of a deep neural network by applying multiple convolution, batch normalization, and ReLU activation operations to the input data dlX.


for k = 1:numel(parameters) 
    % Convolution, Batchnormalization and ReLU function 
    [dlY, state(k)] = perceptron(dlY,parameters(k),state(k),isTraining);
end
end