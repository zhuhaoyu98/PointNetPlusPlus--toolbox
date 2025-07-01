function Psamp=func_samples_layers(P);

global nSAMP;
%Sampling layer.
Psamp      = P;
[Rs,~,~,~] = size(P);
N          = Rs; 
Nsamples   = nSAMP;
%For improved precision, replaced original PointNet sampling with full sampling.
%Randomly select a point.
nums    = 0;
while nums<=N/Nsamples-1;
  
    nums=nums+1;
    if mod(nums,100)==1
       disp('pointNet++ training samples_layer');  
    end
    if nums==1
       idx     = floor(N*rand)+1;
 
       Psel    = P(idx,:,:,:);
    else
       %Select the farthest point as the next chosen point.
       dist0   = sum((P-Psel).^2,4);
       dist1   = sum(dist0,3);
       dist2   = sum(dist1,2);
       [V,I]   = max(dist2);

       
       Psel    = P(I,:,:,:);
    end
     
    Psamp(nums,:,:,:) = [Psel];
end

end

