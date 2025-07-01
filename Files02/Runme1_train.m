clc;
clear;
close all;
warning off;
addpath(genpath(pwd));   % Add all subfolders in the current folder to the function call directory
rng('default')  %%% Initialize random number state

 % Define global variable Dim, representing data dimension
global Dim;
Dim = 3; % Set data dimension

global KKK;
KKK = 10;

global nSAMP;
nSAMP = 100;

global Dim;
Dim = 3; % Set data dimension

% Define sampling number
numPoints = 100;  % Number of points per point cloud sample

% Define global variables Xdim4 and Xtdim4 for storing high-dimensional data
global Xdim4;
global Xtdim4;
% Define global variable istrain to indicate whether it is in training mode
global istrain;

% Create training dataset
dsTrain = PtCloudClassificationDatastore('train_big'); % Create training dataset from 'train_big' directory
% Create validation dataset
dsVal = PtCloudClassificationDatastore('val_big'); % Create validation dataset from 'val_big' directory

%%
% Transform dataset, extract labels and counts
dsLabelCounts = transform(dsTrain, @(data) {data{2} data{1}.Count}); % Transform dataset, extract labels and counts
labelCounts = readall(dsLabelCounts); % Read all labels and their counts
labels = vertcat(labelCounts{:,1}); % Get all labels
counts = vertcat(labelCounts{:,2}); % Get the count for each label

uniqueLabelsTrain = categories(labels);
labelCountsTrain = countcats(labels);
%
% Transform dataset, extract labels and counts
dsLabelCountsVal = transform(dsVal, @(data) {data{2} data{1}.Count}); % Transform dataset, extract labels and counts
labelCountsVal = readall(dsLabelCountsVal); % Read all labels and their counts
labelsVal = vertcat(labelCountsVal{:,1}); % Get all labels
countsVal = vertcat(labelCountsVal{:,2}); % Get the count for each label

uniqueLabelsVal = categories(labelsVal);
labelCountsVal = countcats(labelsVal);
%
numTrain = numel(dsTrain.Files);
numVal = numel(dsVal.Files);
% Total dataset size
dataSizes = [numTrain, numVal];
categories = {'Training Set', 'Validation Set'};

offset = 0.15;
ysbq = 1:length(uniqueLabelsTrain);
ggbqTrain = ysbq - offset;
ggbqVal = ysbq + offset;


% Plot composite graph
figure;

% Dataset size histogram
subplot(2, 1, 1);
bar(dataSizes, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', categories);
ylabel('Number of Samples');
title('Dataset Size Overview');
grid on;

% Distribution by category
subplot(2,1, 2);
bar(ggbqTrain,labelCountsTrain,'FaceColor','r','BarWidth',0.3);
hold on;
bar(ggbqVal,labelCountsVal,'FaceColor','b','BarWidth',0.3);
xlabel('Classes');
ylabel('Number of Samples');
title('Training and Validation Set Distribution');

xticks(ysbq);
xticklabels(uniqueLabelsTrain);
legend('Trainind Set','Validation Set');

% Set random number generator seed to ensure reproducible results
rng(1); % Set random number generator seed to ensure reproducible results

% Find group index and class name for each category
[G, classes] = findgroups(labels); % Find group index and class name for each category
% Calculate number of observations for each category
numObservations = splitapply(@numel, labels, G); % Calculate number of observations for each category
% Set desired number of observations per class to maximum number of observations
desiredNumObservationsPerClass = max(numObservations); % Set desired number of observations per class to maximum number of observations
% Initialize oversampled file list
filesOverSample = []; % Initialize oversampled file list
% Oversample each category
for i = 1:numel(classes)
    if i == 1 % Get files for the first category
        targetFiles = {dsTrain.Files{1:numObservations(i)}}; % Get files for the first category
    else % Get index of current category files
        targetFiles = {dsTrain.Files{numObservations(i-1)+1:sum(numObservations(1:i))}};
% Get index of current category files
    end
    % Randomly replicate files to achieve desired number of observations
    files = targetFiles; % randReplicateFiles(targetFiles, desiredNumObservationsPerClass);
% Randomly replicate files to achieve desired number of observations
    % Add oversampled files to the list
    filesOverSample = vertcat(filesOverSample, files'); % Add oversampled files to the list
end
dsTrain.Files = filesOverSample; % Update the file list of the training dataset

%%
% Randomly shuffle the order of files in the training dataset
Idxx = randperm(length(dsTrain.Files)); % Generate randomly permuted indices
dsTrain.Files = dsTrain.Files(Idxx); % Reorder files according to randomly permuted indices
dsTrain0 = dsTrain; % Backup original training dataset

% Set mini-batch size for training dataset
dsTrain.MiniBatchSize = 32; % Set mini-batch size for training dataset
dsVal.MiniBatchSize = dsTrain.MiniBatchSize; % Set mini-batch size for validation dataset same as training dataset
dsTrain = transform(dsTrain, @augmentPointCloud); % Augment the training dataset

% Calculate minimum, maximum, and average number of points for each category
minPointCount = splitapply(@min, counts, G); % Calculate minimum number of points for each category
maxPointCount = splitapply(@max, counts, G); % Calculate maximum number of points for each category
meanPointCount = splitapply(@(x) round(mean(x)), counts, G); % Calculate average number of points for each category
stats = table(classes, numObservations, minPointCount, maxPointCount, meanPointCount); % Create statistics table

% Transform dataset, select specified number of points
dsTrain = transform(dsTrain, @(data) selectPoints(data, numPoints)); % Select specified number of points for each sample in the training dataset
dsVal = transform(dsVal, @(data) selectPoints(data, numPoints)); % Select specified number of points for each sample in the validation dataset
dsTrain2 = dsTrain; % Backup transformed training dataset
dsTrain = transform(dsTrain, @preprocessPointCloud); % Preprocess the training dataset
dsVal = transform(dsVal, @preprocessPointCloud); % Preprocess the validation dataset
dsTrain3 = dsTrain; % Backup preprocessed training dataset

% Read filenames from the training dataset
filenames = dsTrain3.UnderlyingDatastores{1, 1}.Files;
if Dim > 3 % Called when the dimension to be extracted is greater than 3, this part needs to be adjusted according to the actual data format, currently processed in ply format
    for j = 1:length(filenames)
        fid = pcread(filenames{j}); % Read point cloud file
        fid2 = selectPoints2(fid, numPoints); % Select specified number of points
        Xdim4{j} = fid2; % Store high-dimensional data
    end
end

% Read filenames from the validation dataset
filenames2 = dsVal.UnderlyingDatastores{1, 1}.Files;
if Dim > 3 % Called when the dimension to be extracted is greater than 3, this part needs to be adjusted according to the actual data format, currently processed in ply format
    for j = 1:length(filenames2)
        fidt = pcread(filenames2{j}); % Read point cloud file
        fidt2 = selectPoints2(fidt, numPoints); % Select specified number of points
        Xtdim4{j} = fidt2; % Store high-dimensional data
    end
end
fid = pcread(filenames{1});
Viewdat=fid.Location;

figure;
plot3(Viewdat(:,1),Viewdat(:,2),Viewdat(:,3),'b.');

grid on
xlabel('x');
ylabel('y');
zlabel('z');

% Initialize input channel size
In_size = Dim; % Input channel size
Hd1_size = [64, 128]; % Set first hidden layer channel size
Hd2_size = 256; % Set second hidden layer channel size
% Initialize input transformation parameters and their state
[parameters.InputTransform, state.InputTransform] = initializeTransform(In_size, Hd1_size, Hd2_size); % Initialize input transformation parameters and their state

% Reset input channel size
In_size = Dim; % Input channel size
Hd_size = [64, 64]; % Set hidden layer channel size
% Initialize shared multi-layer perceptron parameters and their state
[parameters.SharedMLP1, state.SharedMLP1] = initializeSharedMLP(In_size, Hd_size); % Initialize shared multi-layer perceptron parameters and their state

% Set input channel size
In_size = 64; % Input channel size
Hd1_size = [64, 128]; % Set first hidden layer channel size
Hd2_size = 256; % Set second hidden layer channel size
% Initialize feature transformation parameters and their state
[parameters.FeatureTransform, state.FeatureTransform] = initializeTransform(In_size, Hd_size, Hd2_size); % Initialize feature transformation parameters and their state

% Set input channel size
In_size = 64; % Input channel size
Hd_size = 64; % Set hidden layer channel size
% Initialize shared multi-layer perceptron parameters and their state
[parameters.SharedMLP2, state.SharedMLP2] = initializeSharedMLP(In_size, Hd_size); % Initialize shared multi-layer perceptron parameters and their state

% Set input channel size
In_size = 64; % Input channel size
Hd_size = [512, 256]; % Set hidden layer channel size
numClasses = numel(classes); % Get number of classes
% Initialize classifier parameters and their state
[parameters.ClassificationMLP, state.ClassificationMLP] = initializeClassificationMLP(In_size, Hd_size, numClasses); % Initialize classifier parameters and their state

%%
% Set number of training epochs
numEpochs = 15; % Set number of training epochs
% Set learning rate
learnRate = 0.0002; % Set learning rate
% Set L2 regularization coefficient
l2Regularization = 0.2; % Set L2 regularization coefficient
% Set learning rate drop period
learnRateDropPeriod = 5; % Set learning rate drop period
% Set learning rate drop factor
learnRateDropFactor = 0.5; % Set learning rate drop factor

% Set gradient decay factor
gradientDecayFactor = 0.8; % Set gradient decay factor
% Set squared gradient decay factor
squaredGradientDecayFactor = 0.999; % Set squared gradient decay factor
% Initialize average gradient
avgGradients = []; % Initialize average gradient
% Initialize average squared gradient
avgSquaredGradients = []; % Initialize average squared gradient

% Initialize training progress plot
[lossPlotter, trainAccPlotter, valAccPlotter] = initializeTrainingProgressPlot; % Initialize training progress plot

%%
% Get number of classes
numClasses = numel(classes); % Get number of classes
% Initialize iteration count
iteration = 0; % Initialize iteration count
% Start recording training time
start = tic; % Start recording training time

for epoch = 1:numEpochs
    disp(['Epoch: ', num2str(epoch)]); % Display current epoch number
    % Reset training and validation datasets
    reset(dsTrain); % Reset training dataset
    reset(dsVal); % Reset validation dataset

    % Iterate through the dataset until no more data can be read
    while hasdata(dsTrain)
        iteration = iteration + 1; % Increment iteration count
        % Read data
        data = read(dsTrain); % Read next batch of data
        % Create batch
        [XTrain, YTrain] = batchData(data, classes); % Create batch

        istrain = 1; % Set to training mode
        % Calculate model gradients and loss
        [gradients, loss, state, acc] = dlfeval2(@modelGradients, XTrain, YTrain, parameters, state);
% Calculate model gradients and loss
        % Apply L2 regularization
        gradients = dlupdate(@(g, p) g + l2Regularization * p, gradients, parameters); % Apply L2 regularization
        % Update network parameters using Adam optimizer
        [parameters, avgGradients, avgSquaredGradients] = adamupdate(parameters, gradients, avgGradients, avgSquaredGradients, iteration, learnRate, gradientDecayFactor, squaredGradientDecayFactor); % Update network parameters using Adam optimizer
        % Update training progress
        D = duration(0, 0, toc(start), "Format", "hh:mm:ss"); % Calculate elapsed time
        title(lossPlotter.Parent, "Epoch: " + epoch + ", Elapsed: " + string(D)); % Update title of training progress plot
        addpoints(lossPlotter, iteration, double(gather(extractdata(loss)))); % Add loss value to loss plot
        addpoints(trainAccPlotter, iteration, acc); % Add training accuracy to training accuracy plot
        drawnow; % Update figure
    end

    % Create confusion matrix
    cmat = sparse(numClasses, numClasses); % Initialize confusion matrix
    % Classify validation data to monitor training process
    while hasdata(dsVal)
        data = read(dsVal); % Read next batch of data
        % Adjust input data dimension
        [XVal, YVal] = batchData(data, classes); % Create batch

        istrain = 0; % Set to validation mode
        % Calculate label prediction
        isTrainingVal = 0; % Set to validation mode
        YPred = pointnetClassifier(XVal, parameters, state, isTrainingVal); % Calculate label prediction

        % Select prediction with highest score as class label
        [~, YValLabel] = max(YVal, [], 1); % Select true label
        [~, YPredLabel] = max(YPred, [], 1); % Select predicted label
        cmat = aggreateConfusionMetric(cmat, YValLabel, YPredLabel); % Update confusion matrix
    end
    % Update validation accuracy plot
    acc = sum(diag(cmat)) ./ sum(cmat, "all"); % Calculate validation accuracy
    addpoints(valAccPlotter, iteration, acc); % Add validation accuracy to validation accuracy plot
    % Update learning rate
    if mod(epoch, learnRateDropPeriod) == 0
        learnRate = learnRate * learnRateDropFactor; % Update learning rate
    end
    % Reset training dataset
    reset(dsTrain); % Reset training dataset
    % Shuffle data every epoch
    dsTrain.UnderlyingDatastore.Files = dsTrain.UnderlyingDatastore.Files(randperm(length(dsTrain.UnderlyingDatastore.Files))); % Shuffle the order of files in the training dataset
    reset(dsVal); % Reset validation dataset
end


save PNET.MAT