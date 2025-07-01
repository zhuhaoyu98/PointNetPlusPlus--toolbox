function [matrix4]= d3tom4(a,cutvalue, gridspace)


%  transform 3D matlab array data into n*4 matrix, where n is the number of points;
%  'a' is the input 3D matlab array data;
%  'cutvalue' makes that  a point's intensity which is lower than cutvalue (meaningless) will be set to 0 and removed/not accounted.
%  'gridspace' is the grid or the real distance between two points in the array, default =1; 
%  first 3 columns of 'matrix4' is the 3d coordinates and the 4th is intensity.

matrix4=[];
[mm,nn,pp]=size(a);

for iii=1:mm
    for jjj=1:nn
        for kkk=1:pp
  if a(iii,jjj,kkk)>=cutvalue
      matrix4=[matrix4;iii*gridspace,jjj*gridspace,kkk*gridspace,a(iii,jjj,kkk)];
  end
      end
    end
end
