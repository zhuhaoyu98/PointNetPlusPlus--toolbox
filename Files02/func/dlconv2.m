function X = dlconv2(X, weights, bias,nvArgs)

%Utilize MATLAB's built-in convolution function as a deep learning module.
%DLCONV  Deep learning convolution 
%
%    Y = DLCONV(X,WEIGHTS,BIAS) computes the deep learning convolution of
%    the input data X using filters defined by WEIGHTS and adds a constant
%    BIAS. X must be a formatted dlarray. Inputs WEIGHTS and BIAS can be
%    formatted dlarrays, unformatted dlarrays, or numeric arrays. Output Y
%    is a formatted dlarray with the same dimension labels as X.
%
%    If WEIGHTS is an unformatted dlarray or a numeric array, then its size 
%    depends on the type of convolution:
%    -  For standard convolution, WEIGHTS must have numSpatialDimensions + 2 
%       dimensions, where numSpatialDimensions is the number of spatial 
%       dimensions of X. The first numSpatialDimensions correspond to the 
%       spatial dimensions of the filters and have matching sizes. The next 
%       two dimensions correspond to the channels and filters, respectively, 
%       and have matching sizes. 
%    -  For grouped convolution, WEIGHTS must have an extra trailing
%       dimension that corresponds to the groups. The size of the channel 
%       dimension must match the number of channels in the input data 
%       divided by the number of groups.
%    The size of the 'C' dimension of output Y is the product of the number 
%    of filters per group and number of groups.
%
%    If WEIGHTS is a formatted dlarray, then it can have multiple dimensions
%    labeled 'S' (spatial), one dimension labeled 'C' (channel), up to one 
%    dimension labeled 'T' (time), and up to two dimensions labeled 'U' 
%    (unspecified). WEIGHTS must have up to as many 'S' dimensions as X. If 
%    WEIGHTS has fewer 'S' dimensions than X, then the function 
%    automatically adds singleton 'S' dimensions to match the number of 'S' 
%    dimensions of X. If WEIGHTS has a 'T' dimension, then X must also
%    contain a 'T' dimension.
%
%    The format of WEIGHTS depends on the type of convolution:
%    -  For standard convolution, WEIGHTS must have one 'C' dimension and  
%       at most one 'U' dimension. The size of the 'C' dimension must match 
%       the size of the 'C' dimension of input X. The size of 'U' dimension 
%       corresponds to the number of filters. If WEIGHTS specifies a single 
%       filter, then you can omit the 'U' dimension. The size of the 'C' 
%       dimension of output Y is equal to the number of filters.
%    -  For grouped convolution, WEIGHTS must have one 'C' dimension and 
%       two 'U' dimensions. The size of the 'C' dimension of WEIGHTS must 
%       be equal to the size of the 'C' dimension of input X divided by the 
%       number of groups. The first 'U' dimension specifies the number of 
%       filters per group. The second 'U' dimension specifies the number of 
%       groups. The size of the 'C' dimension of output Y is the product of 
%       the number of filters per group and the number of groups.
%
%    BIAS can be a numeric array or a formatted or unformatted dlarray with
%    up to one non-singleton dimension. If BIAS is a scalar or has only
%    singleton dimensions, the same bias is applied to each output.
%    Otherwise, the number of elements must match the number of filters
%    specified by WEIGHTS. If BIAS has any dimension labels, the non-
%    -singleton dimension must be labeled 'C'.
%
%    Y = DLCONV(X,WEIGHTS,BIAS,'DataFormat',FMT) computes the deep learning
%    convolution of an unformatted dlarray or a numeric array X. FMT
%    specifies the dimension labels of the input X. If the input X is a
%    numeric array, at least one of WEIGHTS or BIAS must be a dlarray. The
%    output Y is an unformatted dlarray with the same dimension order as X.
%
%    Y = DLCONV(_,'PARAM1',VAL1,'PARAM2',VAL2,...) specifies optional
%    parameter name/value pairs:
%
%      'Stride'                  - Step size for traversing the input,
%                                  specified as one of the following
%                                  - a scalar, in which case the same value
%                                    is used for all directions.
%                                  - a vector [s1,...,sd] where si is
%                                    the stride along the ith dimension.
%                                  The default value is 1.
%      'DilationFactor'          - Step size for sampling the input or
%                                  equivalently the up-sampling factor of
%                                  the filter. Corresponds to an effective
%                                  filter size of filterSize +
%                                  + (filterSize-1) * (dilationFactor-1).
%                                  Specified as one of the following
%                                  - a scalar, in which case the same value
%                                    is used for all directions.
%                                  - a vector [c1,...,cd] where ci is the
%                                    dilation along the ith dimension.
%                                  The default value is 1.
%      'Padding'                 - Padding applied to the input along
%                                  the edges, specified as one of
%                                  the following:
%                                  - 'same'. Padding is set so that
%                                    the output size is the same as
%                                    the input size when the stride is 1.
%                                    Each output edge size is
%                                    ceil(inputSize/stride), where inputSize 
%                                    is the size of the corresponding input
%                                    edge.
%                                  - 'causal'. Apply padding along the
%                                    left edge of the input, with size
%                                    (filterSize - 1) .* DilationFactor.
%                                    This option supports convolving over 
%                                    a single dimension only.
%                                  - Scalar value, in which case the same 
%                                    value is used for all edges.
%                                  - Vector [p1,...,pd] where pi is the
%                                    padding applied to the start and end
%                                    along the ith edge.
%                                  - Matrix [low1,...,lowd;high1,...,
%                                    highd] of size 2-by-d where lowi and
%                                    highi are the padding along the lower
%                                    and upper sides of the ith edge.
%                                  The size of the padding must be less
%                                  than the filter size for each direction.
%                                  The default value is 0.
%      'PaddingValue'            - Value used to pad the input along the
%                                  edges, specified as one of the
%                                  following values: 
%                                  - Scalar value - Pad using the  
%                                    specified value. 
%                                  - 'symmetric-include-edge' - Pad using 
%                                    mirrored input array, including the 
%                                    edge values.
%                                  - 'symmetric-exclude-edge' - Pad using 
%                                    mirrored input array, excluding the 
%                                    edge values.
%                                  - 'replicate' - Pad using repeated 
%                                    input array border elements.
%                                  The default value is 0.
%      'DataFormat'              - Dimension labels of the input data X,
%                                  specified as a char array or string.
%                                  Required if X is not a formatted dlarray.
%      'WeightsFormat'           - Dimension labels of the weights. 
%                                  Use this option when convolving over a 
%                                  'T' dimension and WEIGHTS is an 
%                                  unformatted dlarray or numeric array.
%
% Example 1:
%     % Create input data as formatted dlarray with 3 channels
%     X = dlarray(ones([64 64 3]),'SSC');
%
%     % Create weights as formatted dlarray representing 96 filters
%     % with height and width of 11
%     weights = dlarray(ones([11 11 3 96]),'SSCU');
%
%     % Compute deep learning convolution
%     dlconv(X, weights, 0);
%
% Example 2:
%     % Create input data as formatted dlarray representing a batch of 7
%     % 64-by-64 activations with 8 channels
%     X = dlarray(ones([64 64 8 7]),'SSCB');
%
%     % Create weights as numeric array representing 2 groups of filters
%     % with 3 filters per group
%     weights = ones([5 5 4 3 2]);
%
%     % Compute deep learning convolution
%     dlconv(X,weights,0);
%
% Example 3:
%     % Create input data as numeric array representing a batch of 7
%     % 64-by-64 activations with 8 channels
%     X = ones([64 64 8 7]);
%
%     % Create weights as formatted dlarray representing 8 groups of
%     % filters with 1 filter per group
%     weights = dlarray(ones([5 5 1 1 8]),'SSCUU');
%
%     % Compute deep learning convolution
%     dlconv(X,weights,0,'DataFormat','SSCB');
%
% Example 4:
%     % Create input data as formatted dlarray representing a batch of 7
%     % temporal activations of size 20, with 8 channels.
%     X = dlarray(ones([8 7 20]),'CBT');
%
%     % Create weights as numeric array representing 96 filters of size 5
%     weights = ones([5 8 96]);
%
%     % Compute deep learning convolution, specifying the 'WeightsFormat'
%     % option to carry out temporal convolution
%     dlconv(X,weights,0,'WeightsFormat','TCU');
%
%   See also BATCHNORM, DLARRAY, DLTRANSPCONV, FULLYCONNECT, MAXPOOL, RELU

%   Copyright 2019-2021 The MathWorks, Inc.

arguments
    X
    weights
    bias
    nvArgs.DataFormat {deep.internal.dlarray.mustBeDataFormatType} = ''
    nvArgs.Stride {mustBeNumeric, mustBePositive, mustBeInteger, mustBeNonempty} = 1
    nvArgs.DilationFactor {mustBeNumeric, mustBePositive, mustBeInteger, mustBeNonempty} = 1
    nvArgs.Padding {deep.internal.dlarray.mustBeNonDlarray} = 0
    nvArgs.PaddingValue {deep.internal.dlarray.mustBeNonDlarray} = 0
    nvArgs.WeightsFormat {deep.internal.dlarray.mustBeDataFormatType} = ''
end
global KK;
global Dim;
[RR,CC,KK,DD] = size(X);
 
% Ensure X is a dlarray and include DataFormat
[X, permFormat] = deep.internal.dlarray.validateDataFormatArg(X, nvArgs.DataFormat);

% Get labels of X
labelsX = dims(X);

% Validate the format of the weights
[weights, numSpatialDimsInWeights, convolveOverTime, isWeightsDlarray, isWeightsLabeled] = deep.internal.dlarray.validateWeightsFormat(weights, nvArgs.WeightsFormat);

% Get information about dimensions to convolve over
[convolutionDimsIdx, numConvolutionDims, numSpatialDimsToConvolve, numSpatialDimsInX] =  deep.internal.dlarray.getConvolutionDimensions(labelsX, numSpatialDimsInWeights, convolveOverTime);

% Locate channel dimension and find number of channels of X
isChannelDimInX = any(labelsX=='C');
if isChannelDimInX
    nbChannelsInX = size(X, numSpatialDimsInX+1);
else
    % A singleton C dimension is assumed
    nbChannelsInX = 1;
end

% Extract and validate weights
[weights, permWeights, filterSize, nbChannelsPerGroup, nbFiltersPerGroup, nbGroups] = ...
    deep.internal.dlarray.extractConvolutionFilter(weights, ...
    isWeightsDlarray, numConvolutionDims, numSpatialDimsInX, numSpatialDimsInWeights, convolveOverTime);

% Check that the channel dimension in the weights matches that of the input
% data
deep.internal.dlarray.validateConvolutionChannelDimension(numConvolutionDims+1, numConvolutionDims+3, isChannelDimInX, ... 
    nbChannelsInX, nbChannelsPerGroup, nbFiltersPerGroup, nbGroups, isWeightsLabeled);


% Extract and validate bias
bias = deep.internal.dlarray.extractConvolutionBias(bias, numConvolutionDims, ...
    numConvolutionDims+2, nbFiltersPerGroup*nbGroups, isWeightsLabeled);

% Validate Name-Value pairs and convert them to canonical values.
sizesConvolutionDimsInX = size(X,convolutionDimsIdx);
[nvArgs, effectiveFilterSize, sizesConvolutionDimsInX] = deep.internal.dlarray.validateConvolutionArguments(nvArgs, ...
    numConvolutionDims, sizesConvolutionDimsInX, filterSize, 'Padding');

% Check the filter size is smaller than the input size
deep.internal.dlarray.validateFilterSizeAgainstPaddedInput(effectiveFilterSize, numConvolutionDims, ...
    sizesConvolutionDimsInX, nvArgs.Padding, isWeightsLabeled);

% Create padding strategy
paddingStrategy = nnet.internal.cnn.layer.padding.createPaddingStrategy(nvArgs.PaddingValue);

internalArgs = {nvArgs.Stride,nvArgs.DilationFactor,nvArgs.Padding,paddingStrategy,nbGroups};

 

if Dim==3
   X = deep.internal.dlarray.convolveData(X, weights, permWeights, bias, internalArgs, numConvolutionDims, numSpatialDimsToConvolve, convolveOverTime, isChannelDimInX);
end
if Dim<3
   if KK<=Dim
      X = deep.internal.dlarray.convolveData(dlarray(X(:,:,1:Dim,:)), weights(:,:,1:Dim,:), permWeights, bias, internalArgs, numConvolutionDims, numSpatialDimsToConvolve, convolveOverTime, isChannelDimInX);
   else
      X = deep.internal.dlarray.convolveData(X, weights, permWeights, bias, internalArgs, numConvolutionDims, numSpatialDimsToConvolve, convolveOverTime, isChannelDimInX);
   end
end
if Dim>3
   if KK<=Dim 
      X = deep.internal.dlarray.convolveData(dlarray(X(:,:,1:Dim,:)), weights(:,:,1:Dim,:), permWeights, bias, internalArgs, numConvolutionDims, numSpatialDimsToConvolve, convolveOverTime, isChannelDimInX);
   else
      X = deep.internal.dlarray.convolveData(X, weights, permWeights, bias, internalArgs, numConvolutionDims, numSpatialDimsToConvolve, convolveOverTime, isChannelDimInX);  
   end
end


% Remove dimensions from the dlarray if it was passed as unformatted.
if ~isempty(nvArgs.DataFormat)
    X = stripdims(X);
    X = ipermute(X, permFormat); % Permutes back in the same order as DataFormat
end
end
