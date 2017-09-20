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
possibilities = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
cSigCombos = zeros(length(possibilities)^2, 3);

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

exampleNum = 1;

for i = 1:length(possibilities)
    C_temp = possibilities(i);
    for j = 1:length(possibilities)
        sigma_temp = possibilities(j);
        model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        cSigCombos(exampleNum, :) = [C_temp, sigma_temp, error];
        exampleNum = exampleNum + 1;
    end
end

[lowestError, lowestErrorIndex] = min(cSigCombos(:,3));
C = cSigCombos(lowestErrorIndex, 1);
sigma = cSigCombos(lowestErrorIndex, 2);

% =========================================================================

% After running the above code, I received the results C = 1, j = 0.1
% To avoid the computationally expensive training process running every time,
% I set the variables to these values below and comment out the above code

%C = 1;
%j = 0.1;

end
