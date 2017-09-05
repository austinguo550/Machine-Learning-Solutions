function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Get objective
z = X * theta;
h = 1 ./ (1 + e .^ -z);

% Set up theta/x matrices and vectors
thetaZero = theta(1);
thetaRest = theta(2:length(theta));
xZero = X(:,1); % theta(0) is treated differently, so need to split x too
xRest = X(:,2:size(X,2));

% Cost function
J = 1/m * sum(-y.*log(h) - (1-y).*log(1-h)) + lambda/(2*m)*sum(thetaRest.^2);

% gradient
gradientZero = 1/m * (xZero' * (h - y));
gradient = 1/m .* (xRest' * (h-y)) + lambda/m .* thetaRest;
grad = [gradientZero; gradient];


% =============================================================

end
