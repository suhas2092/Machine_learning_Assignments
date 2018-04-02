%	##########	GAUSSIAN DISCRIMINANT ANALYSIS	###########
%	########## this will work only for 2-class classification	#########
%	##########	test data & train data should be provided in separate files #########
%	########## trainY=1 for +ve class, trainY=0 for -ve class 	#########	

function gda()

	trainX=csvread('habermanTrain.csv');
	[m, n]=size(trainX);
	trainY=trainX(:,n);
	trainX(:,n)=[];
	testX=csvread('habermanTest.csv');
	testY=testX(:,n);
	testX(:,n)=[];	
	[mtrain, ntrain]=size(trainX);
	trainmean1=zeros(1,n-1);
	trainmean2=zeros(1,n-1);


	postrain=find(trainY==1);
	negtrain=find(trainY==0);
	%postest=find(testY==1);
	%negtest=find(testY==0);
	class1=trainX(postrain,:);
	class2=trainX(negtrain,:);

	for ii=1:n-1,

		trainmean1(ii)=sum(class1(:,ii))/size(class1,1);		% mu(+,1)
		%trainmean1(2)=sum(class1(:,2))/size(class1,1);		% mu(+,2)	
		trainmean2(ii)=sum(class2(:,ii))/size(class2,1);		% mu(-,1)
		%trainmean2(2)=sum(class2(:,2))/size(class2,1);		% mu(-,2)
	end;	


	trainXcov=cov(trainX);
	trainXsigma=std(trainX);
	covInv=inv(trainXcov);

	%class=[class1 class2];
	%syms x y;
	xmin = min(trainX(:,1)) - 5;
    xmax = max(trainX(:,1)) + 5;
    %ymin = min(trainX(:,2)) - 5;
    %ymax = max(trainX(:,2)) + 5;

	%%%%% ESTIMATING PRIOR PROBABILITIES %%%%%%
	prior(1)=length(postrain)/mtrain; % P(Y=1)
	prior(2)=length(negtrain)/mtrain; % P(Y=0)

	mtest=size(testX,1);
	confMatrix=zeros(mtest,1);

	for ii=1:mtest,
		testingRow=testX(ii,:);
		c=testingRow-trainmean1;
		d=testingRow-trainmean2;

		powerPos=exp(-0.5*(c*covInv*c'));
		powerNeg=exp(-0.5*(d*covInv*d'));

		powerPos=prior(1)*powerPos;		% P(y=1/x)
		powerNeg=prior(2)*powerNeg;		% P(y=0/x)

		
		if(powerPos < powerNeg)
			predictedClass=0;
		else predictedClass=1;
		end;
		actualClass=testY(ii);

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


	%%%%% ESTIMATING PERFORMANCE PARAMETERS  %%%%%%
	sumTP=length(find(confMatrix==1));
	sumFP=length(find(confMatrix==2));
	sumFN=length(find(confMatrix==3));
	sumTN=length(find(confMatrix==4));

	accuracy=(sumTP+sumTN)/(sumTP+sumFP+sumFN+sumTN);
	precision=sumTP/(sumTP+sumFP);
	recall=sumTP/(sumTP+sumFN);
	fMeasure=2*precision*recall/(precision+recall);

    

    %%%%%%% REPORTING PARAMETERS %%%%%%%%%%%%%
    fprintf('no of test datasets :%d\n',mtest);
    fprintf('no of misclassifications :%d\n',sumFP+sumFN);
	fprintf('accuracy :%.3f\n',accuracy);
	fprintf('precision :%.3f\n',precision);
	fprintf('recall/sensitivity :%.3f\n',recall);
	fprintf('F-Measure :%.3f\n',fMeasure);

	trainmean1
	trainmean2
	trainXcov
	%%%%%%% VISUALIZING THE OUTPUT %%%%%%%%%%
	%gA= @(x,y) discriFun([x,y], meanClass1, varClass1);
	%gB= @(x,y) discriFun([x y], meanClass2, varClass2);

	figure(1);
	%trainCovin=inv(trainXcov);
	meanSum=trainmean1+trainmean2;
	meanDiff=trainmean1-trainmean2;
	tempProd1=trainXcov\meanDiff';	%  inv(trainXcov)*meanDiff
	%size(tempProd1')
	%size(meanSum)
	tempProd2=(1/2)*meanSum*tempProd1;
	temp3=log(prior(1)/prior(2));
	const=tempProd2-temp3;

	ymin=(const-(tempProd1(1)*xmin))/tempProd1(2);
	ymax=(const-(tempProd1(1)*xmax))/tempProd1(2);

	gscatter(trainX(:,1), trainX(:,2), trainY,'rgb','osd');
	hold on;
	plot([xmin,ymin],[xmax,ymax],'Color','k');
	%s = warning('off','all');
 	%gplot1 = ezsurf(gA,[xmin,xmax,ymin,ymax]);hold on;
 	%set(gplot1, 'FaceColor', [0 128 128] / 256);   
    %gplot1 = ezsurf(gB,[xmin,xmax,ymin,ymax]);hold on;
    %set(gplot1, 'FaceColor', [128 128 0] / 256);
    %warning(s);
	xlabel('Attr 1');
	ylabel('Attr 2');
	hold off;


	% ##### 	SURFACE PLOT 	##########

	figure(2);
	x1 = -3:.2:3; x2 = -3:.2:3;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean1,trainXsigma);
	F = reshape(F,length(x2),length(x1));
	surf(x1,x2,F);hold on
	caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
	axis([-3 3 -3 3 0 .1]);
	xlabel('x1'); ylabel('x2'); zlabel('Probability Density');

	x1 = -18:0.2:18; x2 = -18:.2:18;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean2,trainXsigma);
	F = reshape(F,length(x2),length(x1));
	surf(x1,x2,F);
	caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
	axis([-6 10 -6 10 0 .1]);

	hold off;

	%	#######		CONTOUR 	#########

	figure(3);
	x1 = -13:.2:13; x2 = -13:.2:13;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean1,trainXsigma);
	F = reshape(F,length(x2),length(x1));
	%mvncdf([0 0],[1 1],trainmean1',trainXsigma);
	contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);hold on
	xlabel('x'); ylabel('y');
	%line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');


	x1 = -13:.2:13; x2 = -13:.2:13;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean2,trainXsigma);
	F = reshape(F,length(x2),length(x1));
	%mvncdf([0 0],[1 1],trainmean1',trainXcov);
	contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);hold on
	xlabel('x'); ylabel('y');
	%line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');


end

