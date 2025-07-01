function [gradients, loss, state, acc] = modelGradients(X, Y, parameters, state)
    % Execute model function.
    % isTraining %%training parameter specifies computation for training or inference. 
    %Dropout is performed during training.
    isTraining = true;
    
    % For pointnetClassifier details, refer to the helper functions below.
    % Input X is training data, Y is labels, parameters are model parameters, state is model state.
    % Returns YPred as predicted output, state as updated model state, and dlT as the feature transformation matrix (if applicable).
    [YPred, state, dlT] = pointnetClassifier(X, parameters, state, isTraining);

    % Add regularization term to ensure feature transformation matrix is near-orthogonal.
    % In this example, dlT has a size of 64x64. You can change this value based on your data and other settings.
    K = size(dlT, 1); % First dimension size of the feature transformation matrix.
    B = size(dlT, 4); % Fourth dimension size of the feature transformation matrix (typically batch size).
    
    % Create an identity matrix of the same size as the batch, repeated K times (assuming K=64 here).
    I = repelem(eye(K), 1, 1, 1, B); % Generate a KxKxB identity matrix.
    % Convert I to dlarray type, specifying dimension labels "SSCB".

    dlI = dlarray(I, "SSCB");
    
    % Compute the mean squared error (MSE) between the dot product of dlT and its transpose, and the identity matrix.

    treg = mse(dlI, dlmtimes(dlT, permute(dlT, [2 1 3 4])));
    factor = 0.0025; % Weight factor for the regularization term. Increasing this value often improves performance.
    
    % Compute the loss function, including cross-entropy loss and the regularization term.
    loss = crossentropy(YPred, Y) + factor * treg;
    
    % Compute parameter gradients with respect to the loss function.
    gradients = dlgradient(loss, parameters);
    
    % Compute training accuracy metric.
    [~, YTest] = max(Y, [], 1); % Get maximum index of true labels.
    [~, YPred] = max(YPred, [], 1); % Get maximum index of predicted labels.
    acc = gather(extractdata(sum(YTest == YPred) ./ numel(YTest))); % Compute accuracy.
end