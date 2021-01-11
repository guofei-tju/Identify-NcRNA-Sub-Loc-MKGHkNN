function [y_pred, y_score] = multilabel_ghknn(train_x, train_y, test_x, k_nn, lammda, gamma, beta, type)
% @Author: ZHH, TJU CS LBCI team, http://lbci.tju.edu.cn/
% @Date: 2021/01/11

% @Input:
%   train_x: training features to calculate different kernels (matrix m*d, m: number of training samples, d: dimension of feature);
%   train_y: corresponding training labels (matrix m*c, c: multiple classes of labels);
%   test_x: testing samples (matrix of n*d, n: number of testing samples);
%   k_nn: parameter of k nearest neighbors;
%   lammda: parameter of L-2 norm regularization term (0.1);
%   gamma: parameter of RBF kernel function;
%   beta: parameter of graph norm regularization term (0.1);
%   type: types of kernel function ('rbf').
%
% @Output:
%   y_pred: predicted labels for testing samples;
%   y_score: real value score for testing samples;


Trainlabels = train_y;
n_class = size(train_y, 2);
n_test = size(test_x,1);


y_pred = zeros(n_test, n_class);
y_score = zeros(n_test, n_class);

for i=1:n_class
    % str = ['class: ', num2str(i)];
    % disp(str)
    tr_y = Trainlabels(:, i);
    [predict_y,~,score_f] = ghknn(train_x, tr_y, test_x, k_nn, lammda, gamma, beta, type);

    y_pred(:, i) = predict_y;
    y_score(:, i) = score_f;

end


