function err = regularisedTestSetError(X,T,l,y)


	[m,n]=size(X);
	err=0.0;
	hyp=zeros(1,m);
	for i=1:m,
		hyp(i)=X(i,:)*T';  % h(x^(i)) = T'*X[i] getting the hypothesis
		err=err+(hyp(i)-y(i))^2;  % h(x^(i)-y(i))^2
	end;
	T=T.^2;
	sumT=sum(T)-T(1)^2;
	err=err+sumT*l;
	err=err/(m);
	%err=sqrt(err);			

end

