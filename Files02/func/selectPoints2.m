function data = selectPoints2(data,numPoints) 
%Output the input point cloud data data adjusted to numPoints.
 
ptCloud = data;
if ptCloud.Count > numPoints % When the number of input points is larger than the number of points to feed into PointNet 
    percentage = numPoints/ptCloud.Count;
    data       = pcdownsample(ptCloud,"random",percentage); % down-sample the point cloud  
else  % replicate the points to increase the number of points
    replicationFactor = ceil(numPoints/ptCloud.Count);
    ind  = repmat(1:ptCloud.Count,1,replicationFactor);
    data = select(ptCloud,ind(1:numPoints));
end 
 
end