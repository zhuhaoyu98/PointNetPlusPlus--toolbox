function evaluateClassificationFull(trueLabels, predictedScores, predictedLabels, classNames)

% === Automatic type handling (compatible with dlarray, gpuArray, int, etc.) ===
trueLabels       = double(gather(extractdata(trueLabels)));
predictedLabels  = double(gather(extractdata(predictedLabels)));
predictedScores  = double(gather(extractdata(predictedScores)));

% === Type normalization (in case of missing auto-conversion) ===
if isa(trueLabels, 'dlarray')
    trueLabels = extractdata(trueLabels);
end
if isa(predictedLabels, 'dlarray')
    predictedLabels = extractdata(predictedLabels);
end
if isa(predictedScores, 'dlarray')
    predictedScores = extractdata(predictedScores);
end

trueLabels       = double(gather(trueLabels));
predictedLabels  = double(gather(predictedLabels));
predictedScores  = double(gather(predictedScores));
classNames       = cellstr(classNames);  % Force conversion to cell array of strings

% === Reshape labels to ensure consistent dimension ===
trueLabels = trueLabels(:);
predictedLabels = predictedLabels(:);

if size(predictedScores, 1) ~= numel(trueLabels)
    predictedScores = predictedScores';  % Transpose if needed
end

numClasses = numel(classNames);
cmat = confusionmat(trueLabels, predictedLabels);
disp('📊 Confusion Matrix (numeric):');
disp(cmat);

% === Confusion matrix visualization ===
trueCat = categorical(trueLabels, 1:numClasses, classNames);
predCat = categorical(predictedLabels, 1:numClasses, classNames);

figure;
confusionchart(trueCat, predCat, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
xlabel('Predicted Class'); ylabel('True Class');

% === Compute metrics for each class ===
precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1score = zeros(numClasses,1);
specificity = zeros(numClasses,1);

for i = 1:numClasses
    TP = cmat(i,i);
    FP = sum(cmat(:,i)) - TP;
    FN = sum(cmat(i,:)) - TP;
    TN = sum(cmat(:)) - TP - FP - FN;

    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    specificity(i) = TN / (TN + FP + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% === AUC calculation (multi-class One-vs-Rest) ===
aucList = zeros(numClasses,1);
figure; hold on;
colors = lines(numClasses);

for i = 1:numClasses
    isPos = (trueLabels == i);
    [X,Y,~,auc] = perfcurve(isPos, predictedScores(:,i), true);
    plot(X, Y, 'LineWidth', 2, 'Color', colors(i,:));
    aucList(i) = auc;
    legendNames{i} = sprintf('%s (AUC = %.2f)', classNames{i}, auc);
end

legend(legendNames, 'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve (One-vs-Rest)');
grid on; axis square;

% === Display metrics table ===
T = table(classNames(:), precision, recall, f1score, specificity, aucList, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_score', 'Specificity', 'AUC'});
disp('📋 Classification Metrics (Per Class):');
disp(T);

% === Macro-averaged metrics ===
avg_precision = mean(precision);
avg_recall = mean(recall);
avg_f1 = mean(f1score);
avg_specificity = mean(specificity);
avg_auc = mean(aucList);

fprintf('\n📌 Macro-Averaged Metrics:\n');
fprintf('Precision: %.4f\n', avg_precision);
fprintf('Recall: %.4f\n', avg_recall);
fprintf('F1-score: %.4f\n', avg_f1);
fprintf('Specificity: %.4f\n', avg_specificity);
fprintf('AUC: %.4f\n', avg_auc);

end