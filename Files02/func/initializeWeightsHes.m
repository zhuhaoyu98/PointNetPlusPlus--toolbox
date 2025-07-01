function x = initializeWeightsHes(sz)%%%Initialize weights.
fanIn = prod(sz(1:3));
stddev = sqrt(2/fanIn);
x = stddev .* randn(sz);
end