function files = randReplicateFiles(files,numDesired)
%Randomly permute ind.
n = numel(files);
ind = randi(n,numDesired,1);
files = files(ind);
end