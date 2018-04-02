function [jval,grad] = costFunction(X,T,y)

    %debug_on_warning(1);
    %debug_on_error(1);

	[m,n]=size(X);
	jval=0.0;
	hyp=zeros(1,m);
	grad=zeros(1,n);
    
	for i=1:m,
		hyp(i)=X(i,:)*T';  % h(x^(i)) = T'*X[i] getting the hypothesis
		jval=jval+(hyp(i)-y(i))^2;  % h(x^(i)-y(i))^2
	end;
	jval=jval/(2*m);

	for j=1:n,
		for i=1:m,
			grad(j)=grad(j)+(hyp(i)-y(i))*X(i,j);  % h(x^(i)-y(i))*x[i][j] = grad(j)
		end;
		grad(j)=grad(j)/m;
	end;

end