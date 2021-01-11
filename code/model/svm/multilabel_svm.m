function [y_pred, y_score] = multilabel_svm(train_x, train_y, test_x, gamma, c)



Trainlabels = train_y;
n_class = size(train_y, 2);
n_test = size(test_x,1);


y_pred = zeros(n_test, n_class);
y_score = zeros(n_test, n_class);

for i=1:n_class
    % str = ['class: ', num2str(i)];
    % disp(str)
    tr_y = Trainlabels(:, i);
    tr_x = zeros(n_test, 1);
    
    model = fitcsvm(train_x, tr_y, 'KernelFunction', 'rbf', 'BoxConstraint', c, 'KernelScale', gamma);
    ScoreSVMModel = fitSVMPosterior(model);
    [predict_y, score_f] = predict(ScoreSVMModel, test_x);
    %[predict_y, score_f] = predict(model, test_x);
    
    %cmd = ['-t 2',' -c ',num2str(c),' -g ',num2str(gamma),' -b 1 -q'];
    %model = svmtrain(tr_y, train_x, cmd);
    %[predict_y, ~, score_f] = svmpredict(tr_x, test_x, model, '-b 1 -q');

    %idx = find(model.Label == 1);
    y_pred(:, i) = predict_y;
    %y_score(:, i) = score_f(:, idx);
    y_score(:, i) = score_f(:, 2);
    
end


