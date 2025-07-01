function [dlY,state,dlT] = pointnetClassifier(dlX,parameters,state,isTraining)
%Expand data dimensions for PointNet++ classification training.
global Dim;
global Xdim4;
global Xtdim4;
global istrain;

if istrain==1
    [R_,C_,K_,D_]   = size(dlX);
    if Dim==3
       XTrain_4dim = dlX;
    end
    if Dim<3
       XTrain_4dim = zeros(R_,C_,Dim,D_); 
       XTrain_4dim = dlX(:,:,1:Dim,:);
    end
    if Dim>3
       XTrain_4dim = zeros(R_,C_,Dim,D_); 
       XTrain_4dim(:,:,1:3,:) = dlX;
       for jj = 1:D_%Expand data. %%This section may need adjustment based on actual data format; currently, for PLY 4D data, no changes are required.

           fid2 = Xdim4{1,jj}.Intensity;
           fid2 = (fid2-min(fid2))/(max(fid2)-min(fid2));
           XTrain_4dim(:,:,4,jj) = fid2;
       end
    end
    XTrain_4dim = dlarray(XTrain_4dim,'SSCB');
end

if istrain==0
    [R_,C_,K_,D_]   = size(dlX);
    if Dim==3
       XTrain_4dim = dlX;
    end
    if Dim<3
       XTrain_4dim = zeros(R_,C_,Dim,D_); 
       XTrain_4dim = dlX(:,:,1:Dim,:);
    end
    if Dim>3
       XTrain_4dim = zeros(R_,C_,Dim,D_); 
       XTrain_4dim(:,:,1:3,:) = dlX;
       for jj = 1:D_%Expand data. %%This section may need adjustment based on actual data format; currently, for PLY 4D data, no changes are required.

           fid2 = Xtdim4{1,jj}.Intensity;
           fid2 = (fid2-min(fid2))/(max(fid2)-min(fid2));
           XTrain_4dim(:,:,4,jj) = fid2;
       end
    end
    XTrain_4dim = dlarray(XTrain_4dim,'SSCB');
end

%size(XTrain_4dim)=1000           1           4          32
%Sampling layer
XTrain_4dims=func_samples_layers(XTrain_4dim);
%Grouping layer：
XTrain_4dimss=func_grouplayers(XTrain_4dims,XTrain_4dim);
%MSG,%Assign different weights based on density configurations.
%Multi-resolution grouping.（MRG)
XTrain_4dimsss  = func_MSGs(XTrain_4dimss,XTrain_4dim);
XTrain_4dimsss  = dlarray(XTrain_4dimsss,'SSCB');


[dlY,state,dlT] = pointnetEncoder(XTrain_4dimsss,parameters,state,isTraining);

% Invoke the classifier.
p = parameters.ClassificationMLP.Perceptron;
s = state.ClassificationMLP.Perceptron;
for k = 1:numel(p) 
     
    [dlY, s(k)] = perceptron(dlY,p(k),s(k),isTraining);
      
    % If training, apply inverted dropout with a probability of 0.3.
    if isTraining
        probability = 0.3; 
        dropoutScaleFactor = 1 - probability;
        % In this example, the size of dropoutMask is 1*1*512*32 or
        % 1*1*256*32. This is the dimension for classification MLP just
        % like fully connected layer in 2D CNN
        dropoutMask = (rand(size(dlY), "like", dlY) > probability ) / dropoutScaleFactor;
        dlY = dlY.*dropoutMask;
    end
    
end
state.ClassificationMLP.Perceptron = s;

% Apply final fully connected and softmax operations.
weights = parameters.ClassificationMLP.FC.Weights;
bias = parameters.ClassificationMLP.FC.Bias;
dlY = fullyconnect(dlY,weights,bias);
dlY = softmax(dlY);
end