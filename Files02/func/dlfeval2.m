function varargout = dlfeval2(fun,varargin)
%DLFEVAL   %Evaluate deep learning functions for automatic differentiation.
%   [Y1, Y2, ... Ym] = DLFEVAL(FUN, X1, X2, ..., Xn) evaluates FUN(X1, X2,
%   ... Xn) with automatic differentiation tracing enabled. The outputs of
%   FUN are assigned to Y1, Y2, ... Ym. Call DLGRADIENT inside of FUN to
%   compute gradients of traced dlarray objects.
%
%   Examples:
%     a = dlarray([0 0]); % point at which to evaluate gradient
%     function [y,grad] = objectiveAndGradient(x)
%         y = x(1).^2 + x(2) - 2;
%         grad = dlgradient(y,x);
%     end
%     [val,grad] = dlfeval(@objectiveAndGradient,a);
%
%     % non-scalar-valued functions
%     function [y,grad] = objectiveAndGradient(x,scaleFactors)
%         y = x.^2;
%         grad = dlgradient(sum(scaleFactors.*y,'all'),x);
%     end
%     a = dlarray([1 2; 3 4]);
%     scaleFactors = dlarray([0 0; 0 1]);
%     [val,grad] = dlfeval(@objectiveAndGradient,a,scaleFactors);
%     % val is dlarray([1 4; 9 16])
%     % grad is dlarray([0 0; 0 8])
%
%   See also: dlarray/dlgradient

%   Copyright 2018-2021 The MathWorks, Inc.


[varargout{1:nargout}] = deep.internal.dlfeval(fun,varargin{:});        
 
    