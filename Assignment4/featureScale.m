% This fuction performs scaling of the features to the range 0<=i<=1.
% feature scaling facilitaes easy convergence


function X = featureScale(x)

   % debug_on_warning(1);
    %debug_on_error(1);
	[m,n]=size(x);
    maxval=zeros(1,n-1);
    minval=zeros(1,n-1);
	
	for j=2:n,		
		maxval(j-1)=max(x(:,j));
		minval(j-1)=min(x(:,j));

	end;

	for j=2:n,
		difference=maxval(j-1)-minval(j-1);
 	  	for i=1:m,
			x(i,j)=(x(i,j)-minval(j-1))/difference;
	  	end;
	end;
	X=x;
    %disp(x);

end

