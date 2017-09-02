function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % theta = theta - alpha * (derivative of J) <- (2* previousCost* X(i))
    % try to use vectorized implementation

    h = X * theta;   % m x 1
    gradient = 1/m * sum((h - y) .* X)   % 1 x 2 % will broadcast for both columns in X? remember.. batch, and need to implement feature scaling after
    theta = theta - alpha .* (gradient') % 2x1 both components of theta will change at same time



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
     J_history(iter)

end

end
