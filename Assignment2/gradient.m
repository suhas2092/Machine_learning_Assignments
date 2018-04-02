% this function contains the main file.
% it implenets simple linear regression

function  gradient()
    
	maxiter=100;
	alpha=.95;  % choosen after randomly trying many values 
	epsilon=1e-5;
	jval=0.0;
	prompt='K value : ';
	k=input(prompt);
	X=load('housing.txt');
    
	[m,n]=size(X);
	bias=ones(m,1);
	X=[bias X];   % adding bias to the X matrix
	[m,n]=size(X);
	Y = X(:,n);
	X(:,n)=[];
	X=featureScale(X);
	X=[X Y];
    [m,n]=size(X);
	Theta=zeros(k,n-1);
	jval=zeros(1,k);
	grad=zeros(k,n-1);
	testerror=zeros(1,k);
	cost=zeros(maxiter,k);

	set=cvpartition(m,'kfold',k);

	for i=1:maxiter,
		for j=1:k,			
			ip=X(training(set,j),:);
			op=ip(:,n);
			ip(:,n)=[];
			[jval(j),grad(j,:)]=costFunction(ip,Theta(j,:),op);
			cost(i,j)=jval(j);			
			if jval(j)<epsilon,
				break;
			end;
			Theta(j,:)=Theta(j,:)-alpha*grad(j,:);   % vectorised implementation 
			testip=X(test(set,j),:);
			testop=testip(:,n);
			testip(:,n)=[];
			testerror(j)=testSetError(testip,Theta(j,:),testop);
		end;
	end;
    
    [minjval,idx]=min(jval);
    yval=cost(:,idx)';
    fprintf('J(w)values \n');
    disp(yval);
    avg_error=mean(cost(:,idx));
    xval=1:maxiter;
    h=figure;
    plot(xval,yval);
    xlabel('iterations');
    ylabel('J(w)');
    title('Convergence Status');
    fprintf('model parameters\n Weight values \n');
    disp(Theta(idx,:)');
    fprintf('alpha value :%0.2f \n',alpha);
    fprintf('model performance :%0.2f\n',avg_error);

end