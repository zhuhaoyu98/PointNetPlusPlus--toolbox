function Y = oneHotEncode(labels,classes)%One-hot encoding.
numObservations = numel(labels);
numCategories = numel(categories(classes));
% labels = labels';
Y = zeros(numCategories, numObservations, 'single');
for c = 1:numCategories
    Y(c,labels==classes(c)) = 1;
end       
end