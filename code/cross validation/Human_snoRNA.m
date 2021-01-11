clear
seed = 1;
rand('seed', seed);
nfolds = 10; 
nruns = 1;

lammda = 0.1;
type = 'rbf';
beta = 0.1;

%file = {'snoRNA', 'lncRNA', 'miRNA'};
file = {'human_snoRNA', 'human_lncRNA', 'human_miRNA'};

feature_snoRNA = {'DNC', 'TNC', 'Kmer1234', 'Kmer4'};
feature_lncRNA = {'TNC', 'Kmer1234', 'Kmer4', 'CKSNAP', 'DNC', 'RCKmer'};
feature_miRNA = {'NAC', 'CKSNAP'};

feature_human_snoRNA = {'Kmer4', 'Kmer1234', 'DNC'};
feature_human_lncRNA = {'Kmer1234', 'RCKmer', 'TNC', 'Kmer4', 'CKSNAP', 'DNC'};
feature_human_miRNA = {'NAC', 'CKSNAP', 'Kmer1234', 'Kmer4', 'RCKmer'};

gamma_list_snoRNA = [1.00000,0.00781,0.00391,0.00024];
gamma_list_lncRNA = [2.00000,0.25000,0.50000,1.00000,2.00000,1.00000];
gamma_list_miRNA = [0.25000,0.00003];

gamma_list_human_snoRNA = [0.000030518,6.10E-05,1.00E+00];
gamma_list_human_lncRNA = [0.125,0.5,0.5,2.50E-01,5.00E-01,1.00E+00];
gamma_list_human_miRNA = [1,2.50E-01,3.05E-05,0.000030518,0.00097656,3.05E-05];

knn_list_RNA = [50,90,45];
knn_list_human = [55,220,50];


for file_index=1:1
    filename = file{file_index};
    path = ['features\',filename,'\',filename,'.mat']; % path of features
    load(path);
    
    pt = [filename];
    disp(pt);
    
    if strcmp(filename, 'human_snoRNA')
        gamma_list = gamma_list_human_snoRNA;
        feature = feature_human_snoRNA;
        k_nn = 55;
    elseif strcmp(filename, 'human_lncRNA')
        gamma_list = gamma_list_human_lncRNA;
        feature = feature_human_lncRNA;
        k_nn = 220;
    elseif strcmp(filename, 'human_miRNA')
        gamma_list = gamma_list_human_miRNA;
        feature = feature_human_miRNA;
        k_nn = 50;
    elseif strcmp(filename, 'snoRNA')
        gamma_list = gamma_list_snoRNA;
        feature = feature_snoRNA;
        k_nn = 50;
    elseif strcmp(filename, 'lncRNA')
        gamma_list = gamma_list_lncRNA;
        feature = feature_lncRNA;
        k_nn = 90;
    elseif strcmp(filename, 'miRNA')
        gamma_list = gamma_list_miRNA;
        feature = feature_miRNA;
        k_nn = 45;
    end
    
    
    n_features = size(feature, 2);
        
    train_x_list = cell(1, n_features);
    for feature_index=1:n_features
        featurename = feature{feature_index};
        train_x_list{1,feature_index} = line_map(eval([filename, '_', featurename]));
    end
    
    n_samples = size(train_x_list{1,1}, 1);
    train_label = multi_label;
    n_class = size(train_label, 2);
    
    Kernels_list = zeros(n_samples, n_samples, n_features);
    for feature_index=1:n_features
        X = train_x_list{1, feature_index};
        Kernels_list(:, :, feature_index) = kernel_RBF(X, X, gamma_list(1, feature_index));
    end
    
    
    weight_v_list = zeros(n_features, n_class);
    for i=1:n_class
        y = train_label(:, i);
        weight_v_list(:, i) = hsic_kernel_weights_norm(Kernels_list, y, 1, 0.01, 0.001);
    end
    
   
    crossval_idx = crossvalind('Kfold', n_samples, nfolds);

    ACC=[];
    AP = [];
    zero_one_loss = [];
    coverage_error = [];
    label_ranking_loss = [];
    ham_loss = [];


    for run=1:nruns

        for fold = 1:nfolds
            
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
            [acc_score] = Accuracy(y_pred.', te_y.');
            [ap_score] = Average_precision(y_score.', te_y.');
            [zo_error] = One_error(y_score.', te_y.');
            [cov_error] = coverage(y_score.', te_y.');
            [rank_loss] = Ranking_loss(y_score.', te_y.');
            [hm_loss] = Hamming_loss(y_pred.', te_y.');


            ACC = [ACC; acc_score];
            AP = [AP; ap_score];
            zero_one_loss = [zero_one_loss; zo_error];
            coverage_error = [coverage_error; cov_error];
            label_ranking_loss = [label_ranking_loss; rank_loss];
            ham_loss = [ham_loss; hm_loss];

        end

    end

    mean_ACC = mean(ACC);
    mean_AP = mean(AP);
    mean_Zero_One_Loss = mean(zero_one_loss);
    mean_Coverage_Error = mean(coverage_error);
    mean_Label_Ranking_Loss = mean(label_ranking_loss);
    mean_Hamming_loss = mean(ham_loss);

    disp('============================================');
    disp(num2str(mean_ACC));
    disp(num2str(mean_AP));
    disp(num2str(mean_Zero_One_Loss));
    disp(num2str(mean_Coverage_Error));
    disp(num2str(mean_Label_Ranking_Loss));
    disp(num2str(mean_Hamming_loss));

    output = [mean_ACC; mean_AP; mean_Zero_One_Loss; mean_Coverage_Error; mean_Label_Ranking_Loss; mean_Hamming_loss; k_nn];


end

function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); 
end

