function data = preprocessPointCloud(data)
    if ~iscell(data)
        data = {data};
    end
    %The preprocessPointCloud function preprocesses point cloud data by scaling the coordinates (x, y, z) of each point to be between 0 and 1.
    numObservations = size(data,1);
    for i = 1:numObservations
        % Scale points between 0 and 1.

%         data{i,1}

        xlim = data{i,1}.XLimits;
        ylim = data{i,1}.YLimits;
        zlim = data{i,1}.ZLimits;   
        xyzMin = [xlim(1) ylim(1) zlim(1)];
        xyzDiff = [diff(xlim) diff(ylim) diff(zlim)];  
        data{i,1} = (data{i,1}.Location - xyzMin) ./ xyzDiff;

        %data{i,1} = data{i,1}.Location;

    end
end