function err = meanSquareError(X,T,y)

	m=size(X,1);
	err=0;
	for ii=1:m,
		hyp=X(ii,:)*T;
		err=err+(hyp-y(ii))^2;
	end;
	err=err/m;	


end

