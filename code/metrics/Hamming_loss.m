function result=Hamming_loss(Pre_Labels1,test_target1)
%Computing the hamming loss
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(i,j)=1, otherwise Pre_Labels(i,j)=-1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(i,j)=1, otherwise test_target(i,j)=-1
Pre_Labels =  Pre_Labels1';
test_target  = test_target1';

%     [num_labels,num_samples]=size(Pre_Labels);
%    miss_pairs=sum(sum(Pre_Labels~=test_target));
%    HammingLoss=miss_pairs/(num_labels*num_samples);


test_target(test_target ~= 1) = 0;
Pre_Labels(Pre_Labels ~= 1) = 0;
num_samples = size(test_target, 1);
result = 0;
for i = 1:num_samples
    Y_i = test_target(i, :);
    Y_hat_i = Pre_Labels(i, :);
    result_i = nnz(Y_i | Y_hat_i) - nnz(Y_i & Y_hat_i) ;
    if isnan(result_i)  %  NAN
        result_i = 0;
    end
    result = result + result_i;
end
result = result/(14*num_samples);
end

