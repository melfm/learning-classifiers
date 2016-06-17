function [f,g] = logistic_regression_vec(theta, X, y, lambda)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  hypothesis = sigmoid(theta'*X)
  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
  J = (-1/m) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis) );
  
  costRegularizationTerm = lambda/(2*m) * sum( theta(2:end).^2 );

  f = J + costRegularizationTerm;

  % Vectorized Gradient
  g = X * (y- sigmoid(theta'*X))';