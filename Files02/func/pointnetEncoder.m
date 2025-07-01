function [dlY,state,T] = pointnetEncoder(dlX,parameters,state,isTraining)
%Input transformation for point cloud data.
%dataTransform.%The function receives input data dlX with shape [1000, 1, 3, 32], transforms it, and outputs dlY with the same shape [1000, 1, 3, 32].

% [R,C,K,N]=size(dlX);
[dlY,state.InputTransform] = dataTransform(dlX,parameters.InputTransform,state.InputTransform,isTraining);

% Shared MLP.
[dlY,state.SharedMLP1.Perceptron] = sharedMLP(dlY,parameters.SharedMLP1.Perceptron,state.SharedMLP1.Perceptron,isTraining);
% The size of dlY in Line 12 is 1000*1*64*32.
% Consider a certain point (xyz), the xyz data (3-dimensional) was converted into 64 dimensional data.  

% Feature transform.
[dlY,state.FeatureTransform,T] = dataTransform(dlY,parameters.FeatureTransform,state.FeatureTransform,isTraining);
% Input:1000*1*64*32 Output:1000*1*64*32. 

% Shared MLP.
[dlY,state.SharedMLP2.Perceptron] = sharedMLP(dlY,parameters.SharedMLP2.Perceptron,state.SharedMLP2.Perceptron,isTraining);

% Max operation. 
% This process is one of the essential processes in PointNet. The high
% dimensional feature of each points were averaged using max pooling 
dlY = max(dlY,[],1);
end