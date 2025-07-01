function [dlX,dlY,filenames4] = batchDataWithFilenames(data,classes)
%Prepare point cloud data and labels for DL framework, move to GPU.
X = cat(4,data{:,1});
labels = cat(1,data{:,2});
Y = oneHotEncode(labels,classes);
filenames4 = data(:,3);
 
% Cast data to single for processing.
X = single(X);
Y = single(Y);
% Move data to the GPU if possible.
if canUseGPU
    X = gpuArray(X);
    Y = gpuArray(Y);
end
% Return X and Y as dlarray objects.
dlX = dlarray(X,'SCSB');
dlY = dlarray(Y,'CB');
end