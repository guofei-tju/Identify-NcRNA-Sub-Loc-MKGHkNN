function [predict_y, distance_s, score_f] = hknn(train_x,train_y,test_x,k_nn,lammda)


Trainlabels=train_y;
uniqlabels=unique(train_y);
c=max(size(uniqlabels));
n_test = size(test_x,1);

distance_s = zeros(n_test,c);
number_c = zeros(c,1);
for i=1:c
	train_x_c = train_x(find(Trainlabels==uniqlabels(i)),:);
	n_c= size(train_x_c,1);number_c(i) = n_c;
	if k_nn>=n_c
		k_nn_i = n_c;
	else
		k_nn_i = k_nn;
	end
   [k_nn_x_id] = Nearest_Neighbor(k_nn_i,train_x_c,test_x);
   for j=1:n_test
		N_x = train_x_c(k_nn_x_id(j,:),:);
		N_mu = mean(N_x);
		V = N_x-repmat(N_mu,size(N_x,1),1);
		V = V';
		alpha = (V'*V + lammda*eye(k_nn_i))\(V'*(test_x(j,:)-N_mu)');
		dis = norm(test_x(j,:)-N_mu - (V*alpha)');
		distance_s(j,i) = dis;
   
   end
   
end

[maxval,indices]=min(distance_s');
predict_y=uniqlabels(indices);
	sum_ss=distance_s(:,1)+distance_s(:,2);
	score_f=distance_s(:,1)./sum_ss;


end


function [k_nn_x_id] = Nearest_Neighbor(k,xtr,xte)

[n, l] = size(xte);
k_nn_x_id = zeros(n,k);

[r] = euclidean_d(xte,xtr);

	for i=1:size(r,1)
		
		 [c, ii]=sort(r(i,:));
		k_nn_x_id(i,:)=ii(1:k);
	end


end

function [r2] = euclidean_d(X,Y)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r2 = sqrt(r2);
end