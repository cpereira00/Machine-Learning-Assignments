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


%    theta = theta -alpha*(1/m)*((H_theta-y)'*X)'; %((H_theta-y)'*X) deals with both (h(x)-y)x0 and (h(x)-y)x1 giving a 1x2 matrix
%the extra (') on the end (1/m)*((H_theta-y)'*X)' turns my 1x2 matrix to 2x1 matrix, times alpha&m a scalar subtract thta a 2x1

  H_theta = X*theta;
  
  
  theta(1,1) = theta(1) - alpha/m * sum((H_theta-y).*X(:,1));
  theta(2,1) = theta(2) - alpha/m * sum((H_theta-y).*X(:,2));



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
