function [Pnet3]=func_MSGs(PNet2,P0);

%MSG layer. 
Pnet3 = PNet2;
p     = 2;
for i = 1:size(PNet2,1)
    if mod(i,100)==1
       disp('pointNet++ training MSG'); 
    end
    %Detect sample density.
    dist0   = mean((PNet2(i,:)-P0).^2,4);
    dist1   = sum(dist0,3);
    dist2   = extractdata(sum(dist1,2));
    [V,I]   = sort(dist2);%Feature extraction.
 
    imin      = I(1);
    %%Feature aggregation.
    %%Aggregation strategy: constructed based on empirical performance.
    Pnet3(i,:)= 0.95*PNet2(i,:)+0.05*PNet2(imin,:);%*(dmin/dmax)^p
end


