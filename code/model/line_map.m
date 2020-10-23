function [Mapping_X] = line_map(X)


col_x = size(X,2);
Mapping_X = [];
for i=1:col_x
	col_v=[];
    max_ = max(X(:,i));
    min_ = min(X(:,i));
    diff = max_ - min_;
    if diff == 0
        diff = 1;
    end
	col_v = (X(:,i)-min_)/(diff);
	Mapping_X = [Mapping_X,col_v];
end