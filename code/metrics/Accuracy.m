function [ result ] = Accuracy( Y_hat1, Y1)
%CALCACCURACY Summary of this function goes here
%   Detailed explanation goes here
Y_hat = Y_hat1';
Y = Y1';
Y(Y ~= 1) = 0;
Y_hat(Y_hat ~= 1) = 0;

num_samples = size(Y, 1);
result = 0;
for i = 1:num_samples
    Y_i = Y(i, :);
    Y_hat_i = Y_hat(i, :);
    
    result_i = nnz(Y_i & Y_hat_i) /nnz(Y_i | Y_hat_i);  % & 逻辑符号  判断预测结果是否与真实值相同
    if isnan(result_i)  %  NAN 
        result_i = 0;
    end
    result = result + result_i;
end
result = result/num_samples;
end
%ps=0;
%for i=1:300
 %   for j=1:8
  %      if true_Y(i,j)==tes_Y(i,j)
   %         ps=ps+tes_Y(i,j);
    %    end
   % end
%end

