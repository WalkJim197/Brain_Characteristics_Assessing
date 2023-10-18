function AA= Vector2Matrix(n,A) 
%input: n:the dimension of matrix you expect;A:your vector
AA_lower= zeros(n);
index = find( tril(ones(n),-1)==1 );
    for k=1:length(A)
        AA_lower(index(k)) = A(k);
    end
AA_upper=AA_lower';
AA=AA_upper+AA_lower;
end

