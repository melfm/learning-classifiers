function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = 1 / (2 * m) * sum(((X * theta) - y) .^ 2);
Reg_term = (lambda/(2 * m) * sum (theta(2:end).^2));
J = J + Reg_term;

for i = 1 : m
    % Do not penalize theta_0
    grad(1) = grad(1) + (X(i,:) * theta - y(i) ) * X(i,1);
    for j = 2 : size(theta)
        grad(j) = grad(j) + (X(i,:) * theta - y(i)) * X(i,j) + ...
            ((lambda/m) * theta(j));
    end
end

grad = (1/m) * grad;
% =========================================================================

grad = grad(:);

end
