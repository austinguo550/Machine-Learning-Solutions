function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 = 25 x 401
% Theta2 = 10 x 26

% INPUT LAYER
% add bias unit
inputLayer = [ones(m,1) X];   % m x f

% HIDDEN LAYER
% add bias unit
% requires sigmoid
Z2 = Theta1 * inputLayer';  % 25x401 401x5000
hiddenLayer = [ones(1,m)] sigmoid(Z2)];  % 26x5000

% OUTPUT LAYER
% no bias unit
% requires sigmoid
Z3 = Theta2 * hiddenLayer; %10x26 26x5000
outputLayer = sigmoid(Z3);    % 10x5000 (# ex, columns are probs)

[maxProbs classes] = max(outputLayer);
p = classes';
                      
% =========================================================================


end
