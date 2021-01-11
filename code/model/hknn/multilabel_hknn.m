function [y_pred, y_score] = multilabel_hknn(train_x, train_y, test_x, k_nn, lammda)


Trainlabels = train_y;
n_class = size(train_y, 2);
n_test = size(test_x,1);


y_pred = zeros(n_test, n_class);
y_score = zeros(n_test, n_class);

for i=1:n_class
    % str = ['class: ', num2str(i)];
    % disp(str)
    tr_y = Trainlabels(:, i);
    [predict_y,~,score_f] = hknn(train_x, tr_y, test_x, k_nn, lammda);

    y_pred(:, i) = predict_y;
    y_score(:, i) = score_f;

end




