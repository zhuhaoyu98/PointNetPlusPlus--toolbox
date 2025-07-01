%%%Synthesize training and testing data from same-layer files.
function dataPrep(numPoint)
info=dir('*.ply');
names={info.name};

for i=1:numel(names)
    pt=pcread(names{i});
    % randomly rotate the point clouds
    for j=1:50
       percentage=numPoint/pt.Count;
       ptCloudOut = pcdownsample(pt,'random',percentage);
       rotationVector = pi*2 * [rand, rand, rand];
       rotationMatrix = rotationVectorToMatrix(rotationVector);
       ptRot=ptCloudOut.Location*rotationMatrix;
       pcshow(ptRot);drawnow
       pcwrite(pointCloud(ptRot),[(extractBefore(names{i},'.ply')),sprintf('_%d',j),'.ply'])
    end
    
end
end
