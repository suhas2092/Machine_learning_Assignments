

function [jval,grad] = costFunction(X,T,y)

    %debug_on_warning(1);
    %debug_on_error(1);

	[m,n]=size(X);
	jval=0.0;
	hyp=zeros(1,m);
	grad=zeros(1,n);
    
	for ii=1:m,
		hyp(ii)=X(ii,:)*T';  % h(x^(i)) = T'*X[i] getting the hypothesis
		%hyp(ii)
		hyp(ii)=1/(1+exp(-hyp(ii)));
		jval=jval+(y(ii)*log(hyp(ii)))+((1-y(ii))*log(1-hyp(ii)));  % h(x^(i)-y(i))^2
	end;
	jval=-jval/m;

	for jj=1:n,
		for ii=1:m,
			grad(jj)=grad(jj)+(hyp(ii)-y(ii))*X(ii,jj);  % h(x^(i)-y(i))*x[i][j] = grad(j)
		end;
		grad(jj)=grad(jj)/m;
	end;

end

