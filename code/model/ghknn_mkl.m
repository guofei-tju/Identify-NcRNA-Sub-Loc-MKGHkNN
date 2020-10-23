function [predict_y,distance_s,score_f] = ghknn_mkl(train_x_list, train_y, test_x_list, k_nn, lammda, gamma_list, beta, type, weight_v)


Trainlabels = train_y;
uniqlabels = unique(train_y);
c = max(size(uniqlabels));
n_test = size(test_x_list{1,1}, 1);
n_feature = size(test_x_list, 2);


distance_s = zeros(n_test,c);

number_c = zeros(c,1);
for i=1:c
    train_y_c = train_y(find(Trainlabels==uniqlabels(i)),1);
	n_c = size(train_y_c, 1);
    number_c(i) = n_c;
end

for i=1:c
    n_c = number_c(i);
    
    if k_nn>=n_c
		k_nn_i = n_c;
	else
		k_nn_i = k_nn;
    end
    
    train_x_c_list = cell(1, n_feature);
    for f=1:n_feature
        train_x_c_list{1,f} = train_x_list{1,f}(find(Trainlabels==uniqlabels(i)),:);
    end
    
    [k_nn_x_id] = kernel_Nearest_Neighbor(k_nn_i, train_x_c_list, test_x_list, gamma_list, type, weight_v);
    
   for j=1:n_test
       
       K_n_tr = zeros(k_nn_i, k_nn_i);
       K_n_tr_1 = zeros(k_nn_i, 1);
       K_ii = zeros(1, 1);
       
       for f=1:n_feature
           train_x = train_x_list{1, f};
           train_x_c = train_x(find(Trainlabels==uniqlabels(i)),:);
           N_x = train_x_c(k_nn_x_id(j,:),:);
           N_mu = mean(N_x);
           V = N_x-repmat(N_mu,size(N_x,1),1);

           K_n_tr = K_n_tr + kernel_function(V,V,gamma_list(f),type)*weight_v(f,1);
            
           test_x = test_x_list{1, f};
           nc_x = test_x(j,:)-N_mu;
            
           K_n_tr_1 = K_n_tr_1 + kernel_function(V,nc_x,gamma_list(f),type)*weight_v(f,1);
           K_ii = K_ii + kernel_function(nc_x,nc_x,gamma_list(f),type)*weight_v(f,1);
            
       end
       
        L_M = Lap_M_computing(K_n_tr);
        alpha = (K_n_tr + lammda*eye(k_nn_i) + beta*L_M)\(K_n_tr_1);
        dis = sqrt(K_ii - 2*K_n_tr_1'*alpha + alpha'*K_n_tr*alpha);
        distance_s(j,i) = real(dis);

    end
   
end

[maxval, indices] = min(distance_s');
predict_y = uniqlabels(indices);
	sum_ss = distance_s(:,1) + distance_s(:,2);
	score_f = distance_s(:,1)./sum_ss;

end


function [k_nn_x_id] = kernel_Nearest_Neighbor(k, xtr_list, xte_list, gamma_list, type, weight_v)

m = size(xtr_list{1,1}, 1);
n = size(xte_list{1,1}, 1);
n_feature = size(xtr_list, 2);
k_xx_list = zeros(m, m, n_feature);
k_yy_list = zeros(n, n, n_feature);
k_yx_list = zeros(n, m, n_feature);

for f=1:n_feature
    xtr_f = xtr_list{1,f};
    xte_f = xte_list{1,f};
    k_xx_list(:,:,f) = kernel_function(xtr_f, xtr_f, gamma_list(1,f), type);
    k_yy_list(:,:,f) = kernel_function(xte_f, xte_f, gamma_list(1,f), type);
    k_yx_list(:,:,f) = kernel_function(xte_f, xtr_f, gamma_list(1,f), type);
    
end

k_xx = zeros(m, m);
k_yy = zeros(n, n);
k_yx = zeros(n, m);

for f=1:n_feature
    k_xx = k_xx + weight_v(f,1)*k_xx_list(:,:,f);
    k_yy = k_yy + weight_v(f,1)*k_yy_list(:,:,f);
    k_yx = k_yx + weight_v(f,1)*k_yx_list(:,:,f);
end

dist = kernel_distance(k_yy, k_xx, k_yx);

k_nn_x_id = zeros(n,k);


for i=1:size(dist,1)
    [c, ii]=sort(dist(i,:));
    k_nn_x_id(i,:)=ii(1:k);
end

end


function [dist] = kernel_distance(gram_xx, gram_yy, gram_xy)
[m, n] = size(gram_xy);
dist = zeros(m, n);
for i=1:m
    for j=1:n
        dist(i,j) = sqrt(gram_xx(i, i)+gram_yy(j, j) - 2*gram_xy(i,j));
    end
end
end

function [r2] = euclidean_d(X,Y)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r2 = sqrt(r2);
end


function k= kernel_function(X,Y,gamma,type)

if strcmp(type, 'rbf')
	k = kernel_RBF(X,Y,gamma);
elseif strcmp(type,'lap')
	k = kernel_Laplace(X,Y,gamma);
elseif strcmp(type,'liner')
	k = kernel_Liner(X,Y);
elseif strcmp(type,'Poly')
	k = kernel_Polynomial(X,Y,gamma);
elseif strcmp(type,'tanimoto')
	k = kernel_Tanimoto(X, Y);
end


end

%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); % RBF核矩阵
end


%Liner kernel function
function k = kernel_Liner(X,Y)

	k = X*Y'; % 核矩阵
end


%Polynomial kernel function
function k = kernel_Polynomial(X,Y,gamma)
	coef = 0.01;d=2.0;
	k =  (X*Y'*gamma + coef).^d; % 核矩阵
end


%Laplace kernel function
function k = kernel_Laplace(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r = sqrt(r2);
	k = exp(-r*gamma); % 核矩阵
end

function L_M = Lap_M_computing(SS)

num_2 = size(SS,1);

L_M=[];
d = sum(SS);
D = diag(d);
L_D = D - SS;
%Laplacian Regularized
d_tmep=eye(num_2)/(D^(1/2));
L_M = d_tmep*L_D*d_tmep;




end
function k = kernel_Tanimoto(X, Y)
    n_X = size(X, 1);
    n_Y = size(Y, 1);
    k = zeros(n_X, n_Y);
    for i=1:n_X
        for j=1:n_Y
            k(i,j) = tanimoto_similarity(X(i,:), Y(j,:));
        end
    end
end

function ts = tanimoto_similarity(vec_p, vec_q)
    pq = dot(vec_p, vec_q);
    p_square = norm(vec_p).^2;
    q_square = norm(vec_q).^2;
    ts = pq / (p_square + q_square - pq);
end