function ridgeRegresion()

	oldX=load('Data.txt');
	[m,n]=size(oldX);
	Y=oldX(:,n);
	oldX=[ones(m,1) oldX];
    pvec=[2 3 7];
	count=0;
	allData=zeros(60,3);

%%%% CALCULATION OF MEAN SQUARE ERROR WITH DIFF DEGREE POLYNOMIAL  %%%%%
%%%% LAMBDA IS VARIED FROM 0.1 TO 2 %%%%%	
	for ii=1:3,

		%theta=zeros(pvec(ii)+1,1);
		X=powerX(oldX,pvec(ii));

		[m,n]=size(X);		
		set=cvpartition(m,'LeaveOut');


		for l=0.1:0.1:2,

			lambda=l;
			count=count+1;
			trainError=zeros(m,1);
			testError=zeros(m,1);


			for jj=1:m,
				x=X(training(set,jj),:);
				y=x(:,n);
				x(:,n)=[];
				testx=X(test(set,jj),:);
				testy=testx(:,n);
				testx(:,n)=[];
				%size(x'*x)
				%size(eye(pvec(ii)))
				%size(x)				
				theta=pinv(x'*x+lambda*eye(pvec(ii)+1))*x'*y;
				trainError(jj)=meanSquareError(x,theta,y);
				hyp=testx*theta;
				testError(jj)=(hyp-testy)*(hyp-testy);
			end;	

			meanTrainError=mean(trainError);
			meanTestError=mean(testError);
			allData(count,:)=[lambda meanTrainError meanTestError];
		
		end;

	end;	

	%allData

	figure(1);
	plot(0.1:0.1:2,allData(1:20,2),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData(1:20,3),'Color','k');
	legend('Train Error', 'Test Error');
	hold off;					


	figure(2);
	plot(0.1:0.1:2,allData(21:40,2),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData(21:40,3),'Color','k');
	legend('Train Error', 'Test Error');	
	hold off;					


	figure(3);
	plot(0.1:0.1:2,allData(41:60,2),'Color','r');
	xlabel('lambda');
	ylabel('meanTrainError,meanTestError');
	hold on;
	plot(0.1:0.1:2,allData(41:60,3),'Color','k');
	legend('Train Error', 'Test Error');	
	hold off;					


%%%%  CALCULATING THETA WITHOUT LAMBDA  %%%%%
	n=size(oldX,2);
	oldX(:,n)=[];
	%theta=zeros(2,1);
	theta=pinv(oldX'*oldX)*oldX'*Y;

	meanError=meanSquareError(oldX,theta,Y);

	set1=allData(1:20,:);
	set2=allData(21:40,:);
	set3=allData(41:60,:);
	[minset1,idx1]=min(set1(:,3));
	[minset2,idx2]=min(set2(:,3));
	[minset3,idx3]=min(set3(:,3));

	fprintf('best model with 2nd degree polynomial :lambda=%0.2f error estimated=%0.4f\n',set1(idx1,1),minset1);
	fprintf('best model with 3rd degree polynomial :lambda=%0.2f error estimated=%0.4f\n',set2(idx2,1),minset2);
	fprintf('best model with 7th degree polynomial :lambda=%0.2f error estimated=%0.4f\n',set3(idx3,1),minset3);
	fprintf('mean error by least square method :%0.4f\n',meanError);
	size(oldX)
	size(theta)
	yP = oldX*theta;
	figure(4);
	plot(oldX(:,2),Y,'+');
end

