function [best_gamma, best_knn] = grid_multilabel_ghknn(train_x, train_y, lammda, beta, type, crossval_idx)

seed = 1;
rand('seed', seed);
nfolds = 10; 
%nruns = 1;

%lammda = 0.1;
%k_nn = 100;
%type = 'rbf';
%gamma = 2^-6;
%beta = 0.1;

train_label = train_y;

k_nn_l = 10:10:300;
gamma_l = -15:1:10;
gamma_l = 2.^gamma_l;


%ACC_M = zeros(length(gamma_l),length(k_nn_l));
AP_M = zeros(length(gamma_l),length(k_nn_l));
%zero_one_loss_M = zeros(length(gamma_l),length(k_nn_l));
%coverage_error_M = zeros(length(gamma_l),length(k_nn_l));
%label_ranking_loss_M = zeros(length(gamma_l),length(k_nn_l));
%ham_loss_M = zeros(length(gamma_l),length(k_nn_l));

for i=1:length(gamma_l)
    for j=1:length(k_nn_l)
        
    % crossval_idx = crossvalind('Kfold', size(train_x, 1), nfolds);
    
	gamma = gamma_l(i);
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
		train_X_S = train_x(train_idx,:);
		tr_y = train_label(train_idx, :);
		test_X_S = train_x(test_idx, :);
		te_y = train_label(test_idx, :);
		[y_pred, y_score] = multilabel_ghknn(train_X_S, tr_y, test_X_S, k_nn, lammda, gamma, beta, type);
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
    
    s = ['cv(gamma = ', num2str(gamma_l(i)), ', k_nn = ', num2str(k_nn_l(j)), '), AP = ', num2str(mean_AP)];
    disp(s);
	%ACC_M(i,j) = mean_ACC;
    AP_M(i, j) = mean_AP;
    end
end

%max_ACC = max(max(ACC_M));
max_AP = max(max(AP_M));
[inx, iny] = find(AP_M == max_AP);

best_gamma = gamma_l(inx);
best_knn = k_nn_l(iny);

str = ['gamma = ', num2str(best_gamma), ', k_nn = ', num2str(best_knn)];
disp(str);

disp(num2str(max_AP));


