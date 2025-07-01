clc; 
clear; 
close all;
warning off; 
addpath(genpath(pwd));
rng('default');
% Load the trained model
load PNET.MAT
% Define global variable Dim (data dimensionality)
global Dim;
Dim = 3; % Set data dimension
global KKK;
KKK = 10; 
global nSAMP;
nSAMP = 100; 
global Dim;
Dim = 3; % Set data dimension
% Define number of sampled points per point cloud
numPoints = 100; 
% Define global variables Xdim4 and Xtdim4 for storing high-dimensional data
global Xdim4;
global Xtdim4;
% Define global variable istrain to indicate testing mode
global istrain;
% Load external test data
dsTest = PtCloudClassificationDatastore('test_external'); % Create dataset from 'test_external' directory
labels = numel(classes);
testFolder = 'test_external';
files = dir(fullfile(testFolder, '*.ply'));
counts = length(files);
%dsTest.MiniBatchSize = counts; % Set mini-batch size
dsTest = transform(dsTest, @augmentPointCloud); % Augment test dataset
dsTest = transform(dsTest, @(data) selectPoints(data, numPoints)); % Select fixed number of points per sample
dsTest1 = dsTest; % Backup transformed dataset
dsTest = transform(dsTest, @preprocessPointCloud); % Preprocess dataset
dsTest2 = dsTest; % Backup preprocessed dataset
% Read filenames from the validation dataset
filenames = dsTest2.UnderlyingDatastores{1, 1}.Files;
if Dim > 3 % If extracting dimensions > 3, adjust as needed; assumes .ply format
    for j = 1:length(filenames)
        fidt = pcread(filenames{j}); % Read point cloud file
        fidt2 = selectPoints2(fidt, numPoints); % Select fixed number of points
        Xtdim4{j} = fidt2; % Store high-dimensional data
    end
end
% Initialize confusion matrix
cmat = sparse(numel(classes), numel(classes));
reset(dsTest); % Reset test dataset
filenames2 = dsTest.UnderlyingDatastores{1, 1}.Files;
data = readall(dsTest); % Read all test data
data = [data,filenames2];
[XTest, YTest, filenames3] = batchDataWithFilenames(data, classes); % Create batch
filenames3_short = erase(filenames3,'C:\Users\Administrator\Desktop\3.26.1\test_external\'); % Remove common path prefix
%disp(filenames3_short);
% Use pointnetClassifier to classify validation data
YPred = pointnetClassifier(XTest, parameters, state, isTrainingVal); % Predict labels
% Select the label with highest score as predicted class
[~, YPredLabel] = max(YPred, [], 1); 
% Convert predicted numeric labels to class names
YPredClassNames = classes(YPredLabel); % cell array of strings
% Display filenames and their predicted classes
fprintf('\n📄 Filename and Prediction Mapping:\n');
fprintf('-----------------------------------------\n');
fprintf('%-4s %-30s %s\n', 'ID', 'Filename', 'Predicted Class');
fprintf('-----------------------------------------\n');
for i = 1:counts
    [~, name, ext] = fileparts(filenames3_short{i});
    fprintf('%-4d %-30s %s\n', i, [name, ext], YPredClassNames(i));
end
% Prompt for manual input of ground-truth labels
disp("📌 Class label index mapping:");
for i = 1:numel(classes)
    fprintf(" %d = %s\n", i, classes(i));
end
disp("Please input the ground-truth label vector (use the numeric indices shown above), e.g., [1 2 1 3 2 ...]");
trueLabels = input('Enter ground-truth labels: ');
% Validate the number of labels
if numel(trueLabels) ~= counts
    error("❌ Number of input labels (%d) does not match number of samples (%d). Please rerun.", ...
        numel(trueLabels), counts);
end
% Validate label value range
if any(trueLabels < 1) || any(trueLabels > numel(classes))
    error("❌ Labels must be in the range 1 to %d. Please check your input.", numel(classes));
end
% Ensure trueLabels is a double row vector
trueLabels = dlarray(trueLabels);
cmat = aggreateConfusionMetric(cmat, trueLabels, YPredLabel);
acc = sum(diag(cmat)) / sum(cmat, "all");
disp("Overall Accuracy:");
disp(acc);
% Show detailed classification evaluation
%evaluateClassificationMetrics(cmat, classes);
evaluateClassificationFull(trueLabels, YPred, YPredLabel, classes);