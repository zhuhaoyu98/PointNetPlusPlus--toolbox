function data = selectPoints(data,numPoints) 
%Adjust the input point cloud data data to numPoints.
numObservations = size(data,1);% Confirm the number of data
for i = 1:numObservations    
    ptCloud = data{i,1};
    if ptCloud.Count > numPoints % When the number of input points is larger than the number of points to feed into PointNet 
        percentage = numPoints/ptCloud.Count;
        data{i,1} = pcdownsample(ptCloud,"random",percentage); % down-sample the point cloud  
    else  % replicate the points to increase the number of points
        replicationFactor = ceil(numPoints/ptCloud.Count);
        ind = repmat(1:ptCloud.Count,1,replicationFactor);
        data{i,1} = select(ptCloud,ind(1:numPoints));
    end 
end
end