% example01, 
% transform a MATLAB 3D array data to a .ply data with the same name.


%%%% If you have another 3D array to be transformed, just copy the following 6-row codes 
%%%% and update the names of your file (the file name appears 3 times).


%%%% 6-row
clear all
format short
load aa00002;
[matrix4]= d3tom4(aa00002,0.02,0.2);
ptcloud=pointCloud(matrix4(:,1:3),Intensity=matrix4(:,4));
pcwrite(ptcloud,'a00002.ply');
%%%%  end;     


%%  plot point cloud data.
clear all
ptCloud = pcread('a00002.ply')
figure
pcshow(ptCloud)
title("ECD")
xlabel("X")
ylabel("Y")
zlabel("Z")

