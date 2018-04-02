
function  gradient()
    
    %debug_on_warning(1);
    %debug_on_error(1);

	alpha1=0;
	epsilon=1e-5;
	jval=0.0;
	k=4;
	X=csvread('habermanTrain.csv');
    %X(randperm(size(X,1)),:);  % randomize  the elements of matrix X

	[m,n]=size(X);
	bias=ones(m,1);
	X=[bias X];   % adding bias to the X matrix
	[m,n]=size(X);
	%pos=find(Y==1); 
	%neg=find(Y==0);
	Y = X(:,n);
	X(:,n)=[];
	X=featureScale(X);
	%ThetaFixed=ones(1,n-1);
	%plotDecisionBoundary(ThetaFixed, X, Y);
	%X=featureScale(X);
	X=[X Y];
    [m,n]=size(X);
    set1=cvpartition(m,'holdout',0.3);
    XIP=X(training(set1),:);
    TEST=X(test(set1),:);
    m=size(XIP,1);


	set2=cvpartition(m,'kfold',k);
	t=0;
	gmin=zeros(1,10);
	cost1=zeros(1,50);
	allData=zeros(10,2);
	count=1;

	for l=.1:0.1:1,

		t=t+1;
		alpha1=l;
		err=zeros(1,k);


		for ii=1:k,

			ip=XIP(training(set2,ii),:);
			op=ip(:,n);
			ip(:,n)=[];
			tstip=XIP(test(set2,ii),:);
			tstop=tstip(:,n);
			tstip(:,n)=[];
            %size(tstip)
            %size(tstop)
			T=zeros(1,n-1);

			for kk=1:50,
				[jval,grad]=costFunction(ip,T,op);
				if jval<epsilon,
					break;
				end;
				T=T-alpha1*grad;
			end;
			err(ii)=testSetError(tstip,T,tstop);

		end;

		meanError=mean(err);
		allData(count,1)=alpha1;
		allData(count,2)=meanError;
		count=count+1;
	
	end;

	[minjval,idx]=min(allData(:,2));
	alpha1=allData(idx,1);
	%alpha=0.01;

	mtest=size(TEST,1);
	tstop=TEST(:,n);
	TEST(:,n)=[];
	Theta=zeros(1,n-1);

	for kk=1:50,
		[jval,grad]=costFunction(TEST,Theta,tstop);
		cost1(kk)=jval;  %testSetError(TEST,Theta,tstop);;
		if jval<epsilon,
			break;
		end;
		Theta=Theta-alpha1*grad;
	end;

	testX=csvread('habermanTest.csv');
	mtest=size(testX,1);
	testX=[ones(mtest,1) testX];
	testY=testX(:,n);
	testX(:,n)=[];
	testX=featureScale(testX);
	%testX=featureScale(testX);
	confMatrix=zeros(mtest,1);
	ypred=zeros(mtest,1);
	scores=zeros(mtest,1);
	hyp=zeros(mtest,1);	

	for ii=1:mtest,
		pred=Theta*testX(ii,:)';
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


	for ii=1:mtest,
		fprintf('%d %d\n',ypred(ii),testY(ii));
	end;	

	sumTP=length(find(confMatrix==1));
	sumFP=length(find(confMatrix==2));
	sumFN=length(find(confMatrix==3));
	sumTN=length(find(confMatrix==4));

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


	ip=X;
	op=ip(:,n);
	ip(:,n)=[];
	Theta=zeros(1,n-1);

	for jj=1:50,			
		[jval,grad]=costFunction(ip,Theta,op);
		if jval<epsilon,
			break;
		end;
		Theta=Theta-alpha1*grad;   % vectorised implementation
	end;

    fprintf('model parameters\n Weight values\n');
    disp(Theta);
    fprintf('test data error:%0.2f\n',mean(cost1));
    fprintf('alpha value :%f \n',alpha1);
    fprintf('no of test datasets :%d\n',mtest);
    fprintf('no of misclassifications :%d\n',sumFP+sumFN);
	fprintf('accuracy :%.3f\n',accuracy);
	fprintf('precision :%.3f\n',precision);
	fprintf('recall/sensitivity :%.3f\n',recall);
	fprintf('F-Measure :%.3f\n',fMeasure);

	X(:,n)=[];
	plotDecisionBoundary(Theta, X, Y);

end
