function [best_knn] = grid_multilabel_ghknn_mkl(train_x_list, train_y, lammda, beta, type, crossval_idx, gamma_list, weight_v_list, n_features)

seed = 1;
rand('seed', seed);
nfolds = 10; 

train_label = train_y;

k_nn_l = 10:5:200;

%ACC_M = zeros(length(gamma_l),length(k_nn_l));
AP_M = zeros(1, length(k_nn_l));
%zero_one_loss_M = zeros(length(gamma_l),length(k_nn_l));
%coverage_error_M = zeros(length(gamma_l),length(k_nn_l));
%label_ranking_loss_M = zeros(length(gamma_l),length(k_nn_l));
%ham_loss_M = zeros(length(gamma_l),length(k_nn_l));


for j=1:length(k_nn_l)

% crossval_idx = crossvalind('Kfold', size(train_x, 1), nfolds);

k_nn = k_nn_l(j);

%ACC = [];
AP = [];
%zero_one_loss = [];
%coverage_error = [];
%label_ranking_loss = [];
%ham_loss = [];

    for fold=1:nfolds

        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        train_X_S_list = cell(1, n_features);
        test_X_S_list = cell(1, n_features);

        for feature_index=1:n_features
            train_X_S_list{1, feature_index} = train_x_list{1, feature_index}(train_idx,:);
            test_X_S_list{1, feature_index} = train_x_list{1, feature_index}(test_idx, :);
        end

        tr_y = train_label(train_idx, :);
        te_y = train_label(test_idx, :);

        [y_pred, y_score] = multilabel_ghknn_mkl(train_X_S_list, tr_y, test_X_S_list, k_nn, lammda, gamma_list, beta, type, weight_v_list);
        %[acc_score] = Accuracy(y_pred.', te_y.');
        [ap_score] = Average_precision(y_score.', te_y.');
        %[zo_error] = One_error(y_score.', te_y.');
        %[cov_error] = coverage(y_score.', te_y.');
        %[rank_loss] = Ranking_loss(y_score.', te_y.');
        %[hm_loss] = Hamming_loss(y_pred.', te_y.');


        %ACC = [ACC; acc_score];
        AP = [AP; ap_score];
        %zero_one_loss = [zero_one_loss; zo_error];
        %coverage_error = [coverage_error; cov_error];
        %label_ranking_loss = [label_ranking_loss; rank_loss];
        %ham_loss = [ham_loss; hm_loss];

    end
%mean_ACC = mean(ACC);
mean_AP = mean(AP);

s = ['cv(k_nn = ', num2str(k_nn_l(j)), '), AP = ', num2str(mean_AP)];
disp(s);
%ACC_M(1,j) = mean_ACC;
AP_M(1, j) = mean_AP;
end


%max_ACC = max(max(ACC_M));
max_AP = max(max(AP_M));
[~, iny] = find(AP_M == max_AP);

best_knn = k_nn_l(iny);

str = ['k_nn = ', num2str(best_knn)];
disp(str);

disp(num2str(max_AP));


