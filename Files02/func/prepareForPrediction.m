function p = prepareForPrediction(p,fcn)
%The primary purpose of this function is to preprocess data, such as standardization, normalization, or any other necessary transformations, before prediction.
for i = 1:numel(p)
    p(i) = structfun(@(x)invoke(fcn,x),p(i),'UniformOutput',0);
end

    function data = invoke(fcn,data)
        if isstruct(data)
            data = prepareForPrediction(data,fcn);
        else
            data = fcn(data);
        end
    end
end