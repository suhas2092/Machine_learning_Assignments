
function  newgradient()
    
    %debug_on_warning(1);
    %debug_on_error(1);

	epsilon=1e-5;
	jval=0.0;
	k=4;
	X=load('housing.txt');
    %X(randperm(size(X,1)),:);  % randomize  the elements of matrix X

	m=size(X,1);
	bias=ones(m,1);
	X=[bias X];   % adding bias to the X matrix
	n=size(X,2);
	%pos=find(Y==1); 
	%neg=find(Y==0);
	Y = X(:,n);
	%X=powerX(X,7);
	%n=size(X,2);
	X(:,n)=[];
	%ThetaFixed=ones(1,n-1);
	%plotDecisionBoundary(ThetaFixed, X, Y);
	X=featureScale(X);
	X=[X Y];
    [m,n]=size(X);
	ThetaFixed=zeros(1,n-1);

	set1=cvpartition(m,'kfold',k);
	t=0;
	gmin=zeros(1,10);
	optalpha=zeros(1,10);
	cost1=zeros(1,50);

	for l=.1:0.1:1,
		t=t+1;
		alpha1=l;
		%TpermStore=zeros(k,n-1);
		tsterror=zeros(1,k);
		for ii=1:k,
			err=zeros(1,k-1);
			Tstore=zeros(k-1,n-1);
            count=1;
			for jj=1:k,
                %ii
				ip=X(training(set1,jj),:);
				op=ip(:,n);
				ip(:,n)=[];
				tstip=X(test(set1,jj),:);
				tstop=tstip(:,n);
				tstip(:,n)=[];
                %size(tstip)
                %size(tstop)
				T=zeros(1,n-1);

				if(jj~=ii),
					for kk=1:50,
						[jval,grad]=costFunction(ip,T,op);
						if jval<epsilon,
							break;
						end;
						T=T-alpha1*grad;
					end;
					err(count)=testSetError(tstip,T,tstop);
					Tstore(count,:)=T;
                    count=count+1;
				end;
			end;
            %count
            %err
			[~,idx]=min(err);
			errmean=mean(err);
			Tmin=Tstore(idx,:);
            %minjval
			%TpermStore(ii,:)=Tmin;
			%errPerm(ii)=minjval;

			tstip=X(training(set1,ii),:);
			tstop=tstip(:,n);
			tstip(:,n)=[];
			jval=testSetError(tstip,Tmin,tstop);
			tsterror(ii)=errmean;
			%jval
		end;

		minjval=min(tsterror);
		gmin(t)=minjval;
		%minjval
		%alpha
		%gminTheta(t,:)=TpermStore(idx,:);
		%TpermStore(idx,:)
		optalpha(t)=alpha1;
	end;

	[~,idx]=min(gmin);
	alpha1=optalpha(idx);
	%alpha=0.01;

	ip=X;
	%ip(50:100,:)=[];
	op=ip(:,n);
	ip(:,n)=[];

	for jj=1:50,			
		[jval,grad]=costFunction(ip,ThetaFixed,op);
		cost1(jj)=testSetError(ip,ThetaFixed,op);
		%jval			
		if jval<epsilon,
			break;
		end;
		ThetaFixed=ThetaFixed-alpha1*grad;   % vectorised implementation
		%ThetaFixed
		fprintf('%f   ',jval); 
	end

	fprintf('\n');
	costWithoutRegularisation=mean(cost1);

%%%%%  REGULARISATION   %%%%%%
	
	[m,n]=size(X);
	set1=cvpartition(m,'kfold',k);
	allData=zeros(10,2);
	t=1;
	for l=29:0.1:30,
		lambda=l;
		Theta=zeros(1,n-1);
		tsterror=zeros(1,k);
		for ii=1:k,

			ip=X(training(set1,ii),:);
			op=ip(:,n);
			ip(:,n)=[];
			tstip=X(test(set1,ii),:);
			tstop=tstip(:,n);
			tstip(:,n)=[];

			for jj=1:50,
				[jval,grad]=regularizedCostFunction(ip,Theta,lambda,op);
				if jval<epsilon,
					break;
				end;
				Theta=Theta-alpha1*grad;
			end;
			jval=regularisedTestSetError(tstip,Theta,lambda,tstop);
			tsterror(ii)=jval;

		end;

		errmean=mean(tsterror);
		allData(t,1)=lambda;
		allData(t,2)=errmean;
		t=t+1;
		
	end;					

	[~,idx]=min(allData(:,2));
	lambda=allData(idx,1);
	Theta=zeros(1,n-1);
	ip=X;
	op=ip(:,n);
	ip(:,n)=[];
	cost1=zeros(1,50);

	for jj=1:50,			
		[jval,grad]=regularizedCostFunction(ip,Theta,lambda,op);
		cost1(jj)=regularisedTestSetError(ip,Theta,lambda,op);
			%jval			
		if jval<epsilon,
			break;
		end;
		Theta=Theta-alpha1*grad;   % vectorised implementation
		fprintf('%f   ',jval);
	end;


	costWithRegularisation=mean(cost1);

    fprintf('\nmodel parameters\n Weight values(without regularisation) \n');
    disp(ThetaFixed);
    fprintf('model parameters\n Weight values(with regularisation) \n');
    disp(Theta);
    fprintf('cost witout regularisation:%0.2f\n',costWithoutRegularisation);
    fprintf('cost with regularisation:%0.2f\n',costWithRegularisation);    
    fprintf('alpha value :%f \n',alpha1);
    fprintf('model performance :%0.2f\n',jval);
    fprintf('lambda value :%0.2f\n',lambda);

end
