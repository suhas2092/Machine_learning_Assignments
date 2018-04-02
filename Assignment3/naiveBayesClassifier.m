% naive bayes classifier for 2-class classification problem

function  naiveBayesClassifier()

	trainX=csvread('habermanTrain.csv');
	n=size(trainX,2);
	trainY=trainX(:,n);
	trainX(:,n)=[];
	testX=csvread('habermanTest.csv');
	testY=testX(:,n);
	testX(:,n)=[];	
	[mtrain, ntrain]=size(trainX);
	trainmean1=zeros(n-1,1);
	trainmean2=zeros(n-1,1);


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

	%class=[class1 class2];
	%syms x y;
	xmin = min(trainX(:,1));
    xmax = max(trainX(:,1));
    %ymin = min(trainX(:,2)) - 5;
    %ymax = max(trainX(:,2)) + 5;

	%%%%% CONVERT CONTINOUS VALUED ATTRIBUTES TO DISCRETE VALUES %%%%%%
	[disc,classes]=AttributeDiscretizer(trainX,2); 
	phi=zeros(classes,ntrain,2);
	size(phi)
	length(postrain)
	length(negtrain)
	%%%%% ESTIMATING PRIOR PROBABILITIES %%%%%%
	prior(1)=length(postrain)/mtrain; % P(Y=1)
	prior(2)=length(negtrain)/mtrain; % P(Y=0)

	%%%%% ESTIMATION OF PROBABLILITIES OF ALL CLASSES %%%%%%%
	for k=1:classes,
		for jj=1:ntrain,
			indpos=0;
			indneg=0;
			for ii=1:mtrain,

				if(trainY(ii)==1 && disc(ii,jj)==k),
					indpos=indpos+1;	
				elseif(trainY(ii)==0 && disc(ii,jj)==k),
					indneg=indneg+1;
				end;	

			end;
			phi(k,jj,1)=indpos/length(postrain);
			phi(k,jj,2)=indneg/length(negtrain);	



			%%% IF ANY OF THE PHI VALUE IS ZERO GO FOR LAPLACE SMOOTHING  %%%%%%%
			if(phi(k,jj,1)+phi(k,jj,2)==0),
				phi(k,jj,1)=(indpos+1)/(length(postrain)+classes);
				phi(k,jj,2)=(indneg+1)/(length(negtrain)+classes);
			end;

		end;		

	end;
	
	phi
	%discrete=zeros(ntrain,1);
	prob=zeros(ntrain,2);
	mtest=size(testX,1);
	%testingRow=zeros(1,ntrain);
	confMatrix=zeros(mtest,1);
	ypred=zeros(mtest,1);
	scores=zeros(mtest,1);

	for ii=1:mtest,
		testingRow=testX(ii,:);
		discrete=Discretize(testingRow,2); % discrete is a column vector

		for jj=1:ntrain,
			prob(jj,1)=phi(discrete(jj),jj,1);  % P(X=jj/y=1)
			prob(jj,2)=phi(discrete(jj),jj,2);  % P(X=jj/y=0)
			%c=prob(jj,2)
			%prob(3)=phi(discrete(3),1);
		end;

		posProb=prior(1)*prod(prob(:,1));		% P(y=1/x)
		negProb=prior(2)*prod(prob(:,2));		% P(y=0/x)
		fprintf('+VE :%f  -VE :%f\n',posProb,negProb);
		%xprob(2)
		
		if(posProb < negProb)
			predictedClass=0;
		else predictedClass=1;
		end;
		ypred(ii)=predictedClass;
		scores(ii)=posProb;		
		actualClass=testY(ii);

		%%%%% ENTER THE CORRESPONDING VALUE TO CONFUSION MATRIX %%%%%
		%%%%% TP =1, FP=2, FN=3, TN=4  %%%%%%

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
	specificity=sumTN/(sumTN+sumFP);
	figure(4);
    %plot(1-specificity,recall);
    labels=testY;
    [P Q]=perfcurve(labels,scores,1);
    plot(P,Q,'LineWidth',5);


    %%%%%%% REPORTING PARAMETERS %%%%%%%%%%%%%
    fprintf('no of test datasets :%d\n',mtest);
    fprintf('no of misclassifications :%d\n',sumFP+sumFN);
	fprintf('accuracy :%.3f\n',accuracy);
	fprintf('precision :%.3f\n',precision);
	fprintf('recall/sensitivity :%.3f\n',recall);
	fprintf('F-Measure :%.3f\n',fMeasure);


	%%%%%%% VISUALIZING THE OUTPUT %%%%%%%%%%
	%gA= @(x,y) discriFun([x,y], meanClass1, varClass1);
	%gB= @(x,y) discriFun([x y], meanClass2, varClass2);

	figure(1);

	%discrete=Discretize([xmin xmax],2);
	gscatter(trainX(:,1), trainX(:,2), trainY,'rgb','osd');
	hold on;
	%plot([xmin,ymin],[xmax,ymax],'Color','k');
	%s = warning('off','all');
 	%gplot1 = ezsurf(gA,[xmin,xmax,ymin,ymax]);hold on;
 	%set(gplot1, 'FaceColor', [0 128 128] / 256);   
    %gplot1 = ezsurf(gB,[xmin,xmax,ymin,ymax]);hold on;
    %set(gplot1, 'FaceColor', [128 128 0] / 256);
    %warning(s);
	xlabel('Attr 1');
	ylabel('Attr 2');
	hold off;


	figure(2);	%surface plot
	x1 = -3:.2:3; x2 = -3:.2:3;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean1',trainXsigma);
	F = reshape(F,length(x2),length(x1));
	surf(x1,x2,F);hold on
	caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
	axis([-3 3 -3 3 0 .1]);
	xlabel('x1'); ylabel('x2'); zlabel('Probability Density');

	x1 = -18:0.2:18; x2 = -18:.2:18;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean2',trainXsigma);
	F = reshape(F,length(x2),length(x1));
	surf(x1,x2,F);
	caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
	axis([-6 10 -6 10 0 .1]);

	hold off;


	figure(3);	%contour
	x1 = -13:.2:13; x2 = -13:.2:13;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean1',trainXsigma);
	F = reshape(F,length(x2),length(x1));
	%mvncdf([0 0],[1 1],trainmean1',trainXsigma);
	contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);hold on
	xlabel('x'); ylabel('y');
	%line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');


	x1 = -13:.2:13; x2 = -13:.2:13;
	[X1,X2] = meshgrid(x1,x2);
	F = mvnpdf([X1(:) X2(:)],trainmean2',trainXsigma);
	F = reshape(F,length(x2),length(x1));
	%mvncdf([0 0],[1 1],trainmean1',trainXcov);
	contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]);hold on
	xlabel('x'); ylabel('y');
	%line([0 0 1 1 0],[1 0 0 1 1],'linestyle','--','color','k');

end