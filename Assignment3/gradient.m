
function  gradient()
    
    %debug_on_warning(1);
    %debug_on_error(1);

	maxiter=30;
	alpha=0;
	epsilon=1e-5;
	jval=0.0;
	k=4;
	X=csvread('dataTrain.csv');
    %X(randperm(size(X,1)),:);  % randomize  the elements of matrix X

	[m,n]=size(X);
	bias=ones(m,1);
	X=[bias X];   % adding bias to the X matrix
	[m,n]=size(X);
	Y = X(:,n);
	%pos=find(Y==1); 
	%neg=find(Y==0);
	X(:,n)=[];
	%ThetaFixed=ones(1,n-1);
	%plotDecisionBoundary(ThetaFixed, X, Y);
	%X=featureScale(X);
	X=[X Y];
    [m,n]=size(X);
	ThetaFixed=zeros(1,n-1);

	set=cvpartition(m,'kfold',k);
	t=0;
	gmin=zeros(1,10);
	optalpha=zeros(1,10);
	cost=zeros(1,50);

	for l=.1:0.1:1,
		t=t+1;
		alpha=l;
		TpermStore=zeros(k,n-1);
		tsterror=zeros(1,k);
		for ii=1:k,
			err=zeros(1,k-1);
			Tstore=zeros(k-1,n-1);
            count=1;
			for jj=1:k,
                %ii
				ip=X(training(set,jj),:);
				op=ip(:,n);
				ip(:,n)=[];
				tstip=X(test(set,jj),:);
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
						T=T-alpha*grad;
					end;
					err(count)=testSetError(tstip,T,tstop);
					Tstore(count,:)=T;
                    count=count+1;
				end;
			end;
            %count
            %err
			[minjval,idx]=min(err);
			Tmin=Tstore(idx,:);
            %minjval
			TpermStore(ii,:)=Tmin;
			%errPerm(ii)=minjval;

			tstip=X(training(set,ii),:);
			tstop=tstip(:,n);
			tstip(:,n)=[];
			jval=testSetError(tstip,Tmin,tstop);
			tsterror(ii)=jval;
			%jval
		end;

		[minjval,idx]=min(tsterror);
		gmin(t)=minjval;
		%minjval
		%alpha
		%gminTheta(t,:)=TpermStore(idx,:);
		%TpermStore(idx,:)
		optalpha(t)=alpha;
	end;

	[minjval,idx]=min(gmin);
	alpha=optalpha(idx);
	%alpha=0.01;

	ip=X;
	%ip(50:100,:)=[];
	op=ip(:,n);
	ip(:,n)=[];

	for jj=1:50,			
			[jval,grad]=costFunction(ip,ThetaFixed,op);
			cost(jj)=jval;
			%jval			
			if jval<epsilon,
				break;
			end;
			ThetaFixed=ThetaFixed-alpha*grad;   % vectorised implementation
			%ThetaFixed 
	end;

	%cost
	X(:,n)=[];
	plotDecisionBoundary(ThetaFixed, X, Y);
	figure(3);
    plot(1:50,cost);
    xlabel('iterations');
    ylabel('J(w)');
    title('Convergence Status');
	jval=testSetError(X,ThetaFixed,Y);

	testX=csvread('dataTest.csv');
	mtest=size(testX,1);
	testX=[ones(mtest,1) testX];
	testY=testX(:,n);
	testX(:,n)=[];
	%testX=featureScale(testX);
	confMatrix=zeros(mtest,1);
	ypred=zeros(mtest,1);
	scores=zeros(mtest,1);
	hyp=zeros(mtest,1);	

	for ii=1:mtest,
		pred=ThetaFixed*testX(ii,:)';
		if(pred<0),
			predictedClass=0;
		else
			predictedClass=1;
		end;
		actualClass=testY(ii);
		scores(ii)=pred;
		hyp(ii)=1/(1+exp(-pred));		
		ypred(ii)=predictedClass;
		if(predictedClass==1 && actualClass==1)
			confMatrix(ii)=1;
		elseif(predictedClass==1 && actualClass==0)
			confMatrix(ii)=2;	
		elseif(predictedClass==0 && actualClass==1)
			confMatrix(ii)=3;
		elseif(predictedClass==0 && actualClass==0)
			confMatrix(ii)=4;
		end;	

	end;				

	sumTP=length(find(confMatrix==1))
	sumFP=length(find(confMatrix==2))
	sumFN=length(find(confMatrix==3))
	sumTN=length(find(confMatrix==4))

	accuracy=(sumTP+sumTN)/(sumTP+sumFP+sumFN+sumTN);
	precision=sumTP/(sumTP+sumFP);
	recall=sumTP/(sumTP+sumFN);
	fMeasure=2*precision*recall/(precision+recall);
	specificity=sumTN/(sumTN+sumFP);
    labels=testY;
    figure(1);
    [P Q]=perfcurve(labels,scores,1);
    plot(P,Q,'LineWidth',5);

    figure(2);
    plot(scores,hyp);
    fprintf('no of test datasets :%d\n',mtest);
    fprintf('no of misclassifications :%d\n',sumFP+sumFN);
	fprintf('accuracy :%.3f\n',accuracy);
	fprintf('precision :%.3f\n',precision);
	fprintf('recall/sensitivity :%.3f\n',recall);
	fprintf('F-Measure :%.3f\n',fMeasure);


    fprintf('model parameters\n Weight values \n');
    disp(ThetaFixed);
    fprintf('alpha value :%f \n',alpha);
    fprintf('model performance :%0.2f\n',jval);

end
