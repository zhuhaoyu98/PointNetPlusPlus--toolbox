clc;
clear;
close all;
warning off;
addpath(genpath(pwd));   % Add all subfolders in the current folder to the function call directory
rng('default')  %%% Initialize random number state


load PNET.MAT

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
numPoints = 100; % Number of points per point cloud sample

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

folderpath = 'val_big';
plyFiles = dir(fullfile(folderpath,'**','*.ply'));
files1 = plyFiles(~[plyFiles.isdir]);
counts1 = length(files1);

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
        targetFiles = {dsTrain.Files{numObservations(i-1)+1:sum(numObservations(1:i))}}; % Get index of current category files
    end
    % Randomly replicate files to achieve desired number of observations
    files = targetFiles; % randReplicateFiles(targetFiles, desiredNumObservationsPerClass); % Randomly replicate files to achieve desired number of observations
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


cmat = sparse(numClasses, numClasses); % Initialize confusion matrix
% Reset validation dataset
reset(dsVal); % Reset validation dataset 
% Read all validation data
filenames3 = dsVal.UnderlyingDatastores{1, 1}.Files;

data = readall(dsVal); % Read all validation data

data = [data,filenames3];

[XVal, YVal,filenames4] = batchDataWithFilenames(data, classes); % Create batch

filenames4_short = erase(filenames4,'C:\Users\Administrator\Desktop\3.26.1\val_big\'); % Enter the specific file index here
%disp(filenames4_short);


% Classify validation data using pointnetClassifier
YPred = pointnetClassifier(XVal, parameters, state, isTrainingVal); % Calculate label prediction
% Select prediction with highest score as class label
[~, YValLabel]  = max(YVal, [], 1); % Output true value
[~, YPredLabel] = max(YPred, [], 1); % Output predicted value

% Convert predicted numeric labels to class names
YPredClassNames = classes(YPredLabel); % cell array of strings

% Display filenames and their predicted classes
fprintf('\n📄 Filename and Prediction Mapping:\n');
fprintf('-----------------------------------------\n');
fprintf('%-4s %-30s %s\n', 'ID', 'Filename', 'Predicted Class');
fprintf('-----------------------------------------\n');

for i = 1:counts1
    [~, name, ext] = fileparts(filenames4_short{i});
    fprintf('%-4d %-30s %s\n', i, [name, ext], YPredClassNames(i));
end


% Aggregate confusion matrix
cmat = aggreateConfusionMetric(cmat, YValLabel, YPredLabel);

% Calculate and display classification accuracy
acc = sum(diag(cmat)) ./ sum(cmat, "all");
disp('Accuracy:');
disp(acc);

% Evaluate and display additional classification metrics
%evaluateClassificationMetrics(cmat, classes);
evaluateClassificationFull(YValLabel, YPred, YPredLabel, classes);