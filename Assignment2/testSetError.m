function err = testSetError(X,T,y)

    %debug_on_warning(1);
    %debug_on_error(1);

	[m,n]=size(X);
	err=0.0;
	hyp=zeros(1,m);
	for i=1:m,
		hyp(i)=X(i,:)*T';  % h(x^(i)) = T'*X[i] getting the hypothesis
		err=err+(hyp(i)-y(i))^2;  % h(x^(i)-y(i))^2
	end;
	err=err/(2*m);
	err=sqrt(err);			

end
