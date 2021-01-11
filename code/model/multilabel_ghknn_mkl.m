function [y_pred, y_score] = multilabel_ghknn_mkl(train_x_list, train_y, test_x_list, k_nn, lammda, gamma_list, beta, type, weight_v_list)
% @Author: ZHH, TJU CS LBCI team, http://lbci.tju.edu.cn/
% @Date: 2021/01/11

% @Input:
%   train_x_list: list of different training features to calculate different kernels (list of matrix m*d, m: number of training samples, d: dimension of feature);
%       (train_x_list = cell(1, n_features))
%   train_y: corresponding training labels (matrix m*c, c: multiple classes of labels);
%   test_x_list: list of testing samples (list of matrix of n*d, n: number of testing samples);
%   k_nn: parameter of k nearest neighbors;
%   lammda: parameter of L-2 norm regularization term (0.1);
%   gamma_list: list of parameters of RBF kernel function;
%   beta: parameter of graph norm regularization term (0.1);
%   type: types of kernel function ('rbf');
%   weight_v: weights of kernel matrices.
%
% @Output:
%   y_pred: predicted labels for testing samples;
%   y_score: real value score for testing samples;


Trainlabels = train_y;
n_class = size(train_y, 2);
n_test = size(test_x_list{1,1},1);


y_pred = zeros(n_test, n_class);
y_score = zeros(n_test, n_class);

for i=1:n_class
    % str = ['class: ', num2str(i)];
    % disp(str)
    tr_y = Trainlabels(:, i);
    weight_v = weight_v_list(:, i);
    [predict_y,~,score_f] = ghknn_mkl(train_x_list, tr_y, test_x_list, k_nn, lammda, gamma_list, beta, type, weight_v);

    y_pred(:, i) = predict_y;
    y_score(:, i) = score_f;

end


