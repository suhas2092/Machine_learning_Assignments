function newRidge()

	oldX=load('Data.txt');
	[m,n]=size(oldX);
	%Y=oldX(:,n);
	oldX=[ones(m,1) oldX];
    pvec1=[2 3 7 0];
	epsilon=1e-5;    
	%count=0;
	allData1=zeros(20,4);
	allData2=zeros(20,4);
	allData3=zeros(20,4);
	allData4=zeros(20,4);
	count1=1;
	count2=1;	
	count3=1;
	count4=1;	
	set1=cvpartition(m,'holdout',0.3);
    XIP=oldX(training(set1),:);
    Y=XIP(:,n+1);
    TEST=oldX(test(set1),:);
    m=size(XIP,1);

%%%% CALCULATION OF MEAN SQUARE ERROR WITH DIFF DEGREE POLYNOMIAL  %%%%%
%%%% LAMBDA IS VARIED FROM 0.1 TO 2 %%%%%	
	for ii=1:4,

		%theta=zeros(pvec(ii)+1,1);
		if(ii~=4),
			X=powerX(XIP,pvec1(ii));
		end;	
		[m,n]=size(X);
		%X(:,n)=[];
		%X=featureScale(X);
		%X=[X Y];
    	%[m,n]=size(X);
		set=cvpartition(m,'LeaveOut');
		%count=1;

		for l=0.1:0.1:2,

			lambda=l;
			err=zeros(10,1);
			cost2=zeros(10,1);
			count=1;

			for a=0.1:0.1:1,
				alpha1=a;
				trainError=zeros(m,1);
				testError=zeros(m,1);


				for jj=1:m,
					x=X(training(set,jj),:);
					y=x(:,n);
					x(:,n)=[];
					testx=X(test(set,jj),:);
					testy=testx(:,n);
					testx(:,n)=[];
					T=zeros(1,n-1);
					cost1=zeros(50,1);

					for kk=1:50,
						[jval,grad]=regularizedCostFunction(x,T,lambda,y);
						cost1(kk)=jval;
						if jval<epsilon,
							break;
						end;
						T=T-alpha1*grad;
					end;

					testError(jj)=regularisedTestSetError(testx,T,lambda,testy);
					trainError(jj)=mean(cost1);
				end;
	
				cost2(count)=mean(trainError);
				err(count)=mean(testError);
				count=count+1;
				
			end;

			[meanTestError,idx]=min(err);
			meanTrainError=cost2(idx);
			alpha1=0.1*idx;
			if(ii==1),
				allData1(count1,:)=[alpha1 lambda meanTrainError meanTestError];
				count1=count1+1;
			elseif(ii==2),
				allData2(count2,:)=[alpha1 lambda meanTrainError meanTestError];
				count2=count2+1;
			elseif(ii==3),
				allData3(count3,:)=[alpha1 lambda meanTrainError meanTestError];
				count3=count3+1;						
			elseif(ii==4),
				allData4(count4,:)=[alpha1 lambda meanTrainError meanTestError];
				count4=count4+1;
			end;										
	
			
		end;
		
	end;				

	figure(1);
	plot(0.1:0.1:2,allData1(:,3),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData1(:,4),'Color','k');
	legend('Train Error', 'Test Error');
	hold off;					


	figure(2);
	plot(0.1:0.1:2,allData2(:,3),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData2(:,4),'Color','k');
	legend('Train Error', 'Test Error');	
	hold off;					


	figure(3);
	plot(0.1:0.1:2,allData3(:,3),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData3(:,4),'Color','k');
	legend('Train Error', 'Test Error');	
	hold off;					

	[minset1,idx1]=min(allData1(:,4));
	[minset2,idx2]=min(allData2(:,4));
	[minset3,idx3]=min(allData3(:,4));
	[minset4,idx4]=min(allData4(:,4));

	[~,n]=size(TEST);
	Y=TEST(:,n);

	alpha1=allData1(idx1,1);
	lambda=allData1(idx1,2);
	cost1=zeros(50,1);
	X=powerX(TEST,2);
	n=size(X,2);
	X(:,n)=[];
	T=zeros(1,n-1);
	%X=featureScale(X);
	for kk=1:50,
		[jval,grad]=regularizedCostFunction(X,T,lambda,Y);
		cost1(kk)=jval;
		if jval<epsilon,
			break;
		end;
		T=T-alpha1*grad;
	end;
	T
	figure(4);
	plot(1:50,cost1);
	xlabel('iterations');ylabel('J(W)');
	title('Convergence');

	alpha1=allData2(idx2,1);
	lambda=allData2(idx2,2);
	cost1=zeros(50,1);
	X=powerX(TEST,3);
	n=size(X,2);
	X(:,n)=[];
	T=zeros(1,n-1);	
	%X=featureScale(X);
	for kk=1:50,
		[jval,grad]=regularizedCostFunction(X,T,lambda,Y);
		cost1(kk)=jval;
		if jval<epsilon,
			break;
		end;
		T=T-alpha1*grad;
	end;
	T
	figure(5);
	plot(1:50,cost1);
	xlabel('iterations');ylabel('J(W)');
	title('Convergence');

	alpha1=allData3(idx3,1);
	lambda=allData3(idx3,2);
	cost1=zeros(50,1);
	X=powerX(TEST,7);
	n=size(X,2);
	X(:,n)=[];
	T=zeros(1,n-1);	
	X=featureScale(X);
	for kk=1:50,
		[jval,grad]=regularizedCostFunction(X,T,lambda,Y);
		cost1(kk)=jval;
		if jval<epsilon,
			break;
		end;
		T=T-alpha1*grad;
	end;
	T
	figure(6);
	plot(1:50,cost1);
	xlabel('iterations');ylabel('J(W)');
	title('Convergence');

	

	fprintf('best model with 2nd degree polynomial :alpha=%0.2f lambda=%0.2f error estimated=%0.4f\n',allData1(idx1,1),allData1(idx1,2),minset1);
	fprintf('best model with 3rd degree polynomial :alpha=%0.2f lambda=%0.2f error estimated=%0.4f\n',allData2(idx2,1),allData2(idx2,2),minset2);
	fprintf('best model with 7th degree polynomial :alpha=%0.2f lambda=%0.2f error estimated=%0.4f\n',allData3(idx3,1),allData3(idx3,2),minset3);

	fprintf('without regularisation :alpha=%0.2f error estimated=%0.4f\n',allData4(idx4,1),minset4);

end

