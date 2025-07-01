function data = augmentPointCloud(data)
numObservations = size(data,1);

for i = 1:numObservations  
 
 
    ptCloud = data{i,1};   
 
    data{i,1} = ptCloud;
end

 
end

 