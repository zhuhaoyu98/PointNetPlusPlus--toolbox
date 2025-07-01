function [P1]=func_grouplayers(Psamp,P0);
%Grouping layer. 
global KKK;
[RR,CC,KK,NN]=size(Psamp);

P1=zeros(RR,CC,KK,NN);
K = KKK;
for i = 1:size(Psamp,1)/K
    if mod(i,100)==1
       disp('pointNet++ training grouplayer');
    end

    dist0   = sum((Psamp(i,:,:,:)-P0).^2,4);
    dist1   = sum(dist0,3);
    dist2   = extractdata(sum(dist1,2));
    [V,I]   = sort(dist2);
    %Each sub-region contains K points, each with dimension i. After the Grouping layer, the network's output becomes: 
    P1(K*(i-1)+1:K*i,:,:,:) = [Psamp(I(1:K),:,:,:)];
end