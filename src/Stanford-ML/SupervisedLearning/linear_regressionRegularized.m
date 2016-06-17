function [f,g] = linear_regressionRegularized(theta, X,y, lambda)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2); % Number of training examples
  n=size(X,1); % Number of features in each vector X

  f=0;          %Objective or cost function initialize to 0
  g=zeros(size(theta));  %Gradient objective w.r.t. theta

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.

  % Compute 'f' as the objective function as the sum
  for i = 1 : m % For every training example
      % 1/2 ∑i(θ⊤  x(i) −  y(i)) ^ 2
      f = f + ( theta' * X(:,i) - y(i) ) ^2;
  end
  f = f / 2;

  % Do not penalize theta0
  g(1) = g(1) + X(1,1) * (theta' * X(:,1) - y(1));
  % Compute 'g' as the gradient of the objective w.r.t theta
  for i = 2 : m - 1
      for j = 2 : n
          % Sum of X(j,i) * (h(x(i)) - y(i))
          % :  == read column i of all rows
          % ∂J(θ)∂θj=∑ix(i)j  * (hθ(x(i))−y(i))
          g(j) = g(j) + X(j,i) * (theta' * X(:,i) - y(i)) + (lambda/m)*(theta(j));
      end
  end
 

