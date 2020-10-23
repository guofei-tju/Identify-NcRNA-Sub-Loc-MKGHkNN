function [y_pred, y_score] = multilabel_ghknn_mkl(train_x_list, train_y, test_x_list, k_nn, lammda, gamma_list, beta, type, weight_v_list)



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


