function X = featureScale(x)

	[m,n]=size(x);
	maxval=max(max(x));
	minval=min(min(x));
	difference=maxval-minval;
	for i=1:m,
 	  	for j=2:n,
			x(i,j)=(x(i,j)-minval)/difference;
	  	end;
	end;
	X=x;
    
end
