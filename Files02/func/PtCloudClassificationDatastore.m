classdef PtCloudClassificationDatastore < matlab.io.Datastore
    % PtCloudClassificationDatastore Datastore with point clouds
    % and associated categorical labels.
    %
    % ds = PtCloudClassificationDatastore(foldername) constructs a datastore
    % that represents point clouds and associated categories for your dataset. 
    % The input, foldername, is a string or char array
    % which represents the name of the folder that stores trining or test point cloud data.
     %Processes point cloud data and their associated class labels. It allows users to read point cloud data from a specified folder and organize it with corresponding class labels for use in machine learning or deep learning tasks.
    properties
        MiniBatchSize = 1;
    end
    
    properties(Dependent)
        Files
    end
    
    properties(Access = private)
        FileDatastore
    end
    
    methods
        function this = PtCloudClassificationDatastore(datapath)
            % Please put the training and test data folder in your current
            % directly. 
           if nargin==0
               return;
           end
            this.FileDatastore = fileDatastore(datapath,'ReadFcn',@extractTrainingData, ...
            'IncludeSubfolders',true,'FileExtensions',{'.ply','.pcd'});
        end
        
        function tf = hasdata(this)
            tf = hasdata(this.FileDatastore);
        end
        
        function [data,info] = read(this)
            
            if ~hasdata(this)
                error('Reached end of data. Reset datastore.');
            end

            % Preallocate output.
            batchSize = this.MiniBatchSize;
            data = cell(batchSize,3);
            info = struct(...
                'Filename',cell(batchSize,1),...
                'FileSize',cell(batchSize,1));
            
            % Read mini-batch size worth of data. The size of data can be
            % less than the specified batch size.
            idx = 0;
            while hasdata(this.FileDatastore)
                idx = idx + 1;
                [dataOut,infoOut] = read(this.FileDatastore);
                data{idx,1} = dataOut{1};
                data{idx,2} = dataOut{2};
                data{idx,3} = infoOut.Filename;
                info(idx) = infoOut;
                if idx == batchSize
                    break;
                end
            end  
            
            data = data(1:idx,:);
            info = info(1:idx);
            
        end
        
        function reset(this)
            reset(this.FileDatastore);
        end
        
        function files = get.Files(this)
            files = this.FileDatastore.Files;
        end
        
        function set.Files(this,files)
            this.FileDatastore.Files = files;
        end
    end
    
    methods(Access=protected)
        function newds = copyElement(this)
            newds = PtCloudClassificationDatastore();
            newds.FileDatastore = copy(this.FileDatastore);
            newds.MiniBatchSize = this.MiniBatchSize;
%           Shuffle files and corresponding labels
            numObservations = numel(newds.Files);
            idx = randperm(numObservations);
            newds.Files = newds.Files(idx);
        end
    end
   
end

function dataOut = extractTrainingData(fname)

pointData = readbin(fname);
% The label information was obtained using the folder name
idx=strfind(fname,'\');
name=fname(idx(end-1)+1:idx(end)-1);
dataOut = {pointCloud(pointData),categorical(cellstr(name))};
end

function pointData = readbin(fname)
% readbin Read point in the trainig or test files.
pointData = pcread(fname);
pointData =pointData.Location;
end