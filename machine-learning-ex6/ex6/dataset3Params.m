function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

values_C = [0.01 0.03 0.1 0.3 1 3 10 30];
m = size(values_C)(2);
values_sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
n = size(values_sigma)(2);

res = zeros(m, n);

for i = 1:m
  for j = 1:n
    C = values_C(i);
    sigma = values_sigma(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    predict = mean(double(predictions == yval));
    res(i, j) = predict;
  end
end

[M,I] = max(res(:));
[max_i, max_j] = ind2sub(size(res),I);

C = values_C(max_i);
sigma = values_sigma(max_j);

% =========================================================================

end
