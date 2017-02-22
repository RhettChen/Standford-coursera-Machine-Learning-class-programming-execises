function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
yVector = zeros(m ,num_labels);
for i = 1 : m
    yVector(i,y(i)) = 1;
end

onesvec = ones(m,1);
a_layer1 = [onesvec X];
z_layer2 = Theta1 * a_layer1';
a_layer2 = sigmoid(z_layer2);

a_layer2 = [onesvec' ;a_layer2];
z_layer3 = Theta2 * a_layer2;
a_layer3 = sigmoid(z_layer3);

preditAll = a_layer3';
%[maxVal index] = max(preditAll,[],2);
%preditOne = zeros(m,num_labels);
%for i = 1: m
%    preditAll(i,index(i)) = 1;
%end

sumall = 0;
for i = 1 : m
    for k = 1 : num_labels  
        sumall = sumall + yVector(i,k) * log(preditAll(i,k)) + (1 -yVector(i,k)) * log(1 - preditAll(i,k));
    end
end
sumall = sumall * (-1/m);
sumall = sumall + (lambda/(2*m)) * (sum(sum((Theta1.^2),2),1) + sum(sum((Theta2.^2),2),1));
sumall = sumall - (lambda/(2*m)) * (sum(Theta1(:,1).^2) + sum(Theta2(:,1).^2));

J = sumall;

trian_layer1 = zeros(hidden_layer_size,input_layer_size+1);
trian_layer2 = zeros(num_labels,hidden_layer_size+1);
for i = 1 : m 
    differ_layer3 = preditAll(i,:) - yVector(i,:);
    differ_layer3 = differ_layer3';
    differ_layer2 = (Theta2' * differ_layer3) .* sigmoidGradient([1;z_layer2(: , i)]);
    trian_layer1 = trian_layer1 + differ_layer2(2:end) * (a_layer1(i,:));
    trian_layer2 = trian_layer2 + differ_layer3 * (a_layer2(:,i))';
   % Theta1_grad = trian_layer1
end

Theta1_grad = (1/m) .* trian_layer1 + (lambda/m) .* Theta1;
Theta1_grad(: , 1) = Theta1_grad(: , 1) - (lambda/m) .* Theta1(: , 1);

Theta2_grad = (1/m) .* trian_layer2 + (lambda/m) .* Theta2;
Theta2_grad(: , 1) = Theta2_grad(: , 1) - (lambda/m) .* Theta2(: , 1);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
