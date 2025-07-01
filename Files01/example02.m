% example02, 
% transform a .txt data to a .ply data with the same name.


%%%% If you have another .txt to be transformed, just copy the following 6-row codes 
%%%% and update the names of your file (the file name appears 6 times);
clear all
format short
load xd00001.txt;
xd00001(find(xd00001(:,4))<0.2,:)=[];   
ptcloud=pointCloud(xd00001(:,1:3),Intensity=xd00001(:,4));
pcwrite(ptcloud,'xd00001.ply');
%%%%  end;     
 

