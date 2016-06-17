function [f,g] = logistic_regression(theta, X,y)
    %
    % Arguments:
    %   theta - A column vector containing the parameter values to optimize.
    %   X - The examples stored in a matrix.
    %       X(i,j) is the i'th coordinate of the j'th example.
    %   y - The label for each example.  y(j) is the j'th example's label.
    %

    m=size(X,2);
    n=size(X,1); % Number of features in each vector X

    % initialize objective value and gradient.
    f = 0;
    g = zeros(size(theta));

    % Compute cost function
    for i = 1 : m % For every training example
        % -1/m ∑ y(i)(θ⊤  x(i)) + (1 - y(i)) log ( 1 - θ⊤  x(i))
        f = f + ( y(i) * log(sigmoid(X(:,i)' * theta)) + (1 - y(i) * ...
                log(1 - sigmoid(X(:,i)' * theta))));
    end
    f = - f / (m);

    % Compute the gradient
    for i = 1:m,
        h = sigmoid(theta'*X(:,i));
        temp = y(i) - h;
        for j = 1:n,
            g(j) = g(j) + temp * X(j,i);
        end;
    end;

