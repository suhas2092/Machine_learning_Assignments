function som()

	load fisheriris;
	%%  find the no nodes required %%
	%% i used the following heuristic for that %%
	%% 5*m^0.54321 ,m=no of data points. for m=150 it is 76 %%
	grid=zeros(8,8);
	grid_size=8;
	no_of_nuerons=64;
	[m,n]=size(meas);
	X=featureScale(meas);
	%X=meas;
	weight=rand(no_of_nuerons,4);
	epoch=400;
	sigmaO=size(grid,1)/2;  %% radius of the grid
	lambda=epoch/log(sigmaO);
	alphaO=0.05; %0.3
	sigmaT=sigmaO;
	alphaT=alphaO;

	figure(3);
	for kk=1:no_of_nuerons,
		%r=floor(kk/grid_size);
		%c=rem(kk,grid_size);
		%region=grid_size*(r-1)+c;
		subplot(grid_size,grid_size,kk,'align');
		img=reshape(weight(kk,:),2,2)';
		imshow(double(img));
	end;

	%% forming the targeted outputs of the iris data set
	%% NOTE: this part is separately written for iris data set. If you are using any other data set
	%% you can comment this line and set the target vector with the given target values if available
	%% since this is unsupervised learning it is not mandatory to have a target
	%% here we are using the target to analyze the accoracy of the classification
	target=zeros(150,1);
	target(1:50)=1;
	target(51:100)=2;
	target(101:150)=3;

	set1=cvpartition(m,'holdout',0.2);
	trainX=X(training(set1),:);
	trainY=target(training(set1));
	testX=X(test(set1),:);
	testY=target(test(set1));
	mtrain=size(trainX,1);
	%set2=cvpartition(mtrain,'kfold',4);

	%getting the position matrix of all nodes in the grid
	position=zeros(no_of_nuerons,2);
	for kk=1:no_of_nuerons,
		r=floor(kk/grid_size);
		c=rem(kk,grid_size);
		position(kk,1)=r;
		position(kk,2)=c;
	end;	

	%position
	for ii=1:epoch,

		for jj=1:mtrain,
			distval=zeros(no_of_nuerons,1);
			for kk=1:no_of_nuerons,
				distval(kk)=norm(trainX(jj,:)-weight(kk,:));
			end;
			
			% getting the BEST matching unit
			[BMU,idx]=min(distval);

			%updating the weights
			for kk=1:no_of_nuerons,
				distToBmu=norm(position(kk,:)-position(idx,:))^2;				
				%distToBmu=norm(weight(kk,:)-weight(idx,:))^2;
				%sigmaT^2
				%if distance is less than radius,update the weights
				if(distToBmu<(sigmaT^2)),
					nbhdT=exp(-distToBmu/(2*sigmaT^2));
					weight(kk,:)=weight(kk,:)+alphaT*nbhdT*(trainX(jj,:)-weight(idx,:));
				end;	
			end;	

		end;

		%finding the new radius of the neighbourhood
		sigmaT=sigmaO*exp(-(ii/lambda));
		%finding new learning rate
		alphaT=alphaO*exp(-(ii/lambda));

	end;	

	%weight
	clusters=zeros(no_of_nuerons,1);
	clusters=kmeans(weight,3);
	class1=length(find(clusters==1));
	class2=length(find(clusters==2));
	class3=length(find(clusters==3));
	fprintf('fraction of class1:%0.3f\n',class1/no_of_nuerons);
	fprintf('fraction of class2:%0.3f\n',class2/no_of_nuerons);
	fprintf('fraction of class3:%0.3f\n',class3/no_of_nuerons);

	class1=find(clusters==1);
	class2=find(clusters==2);
	class3=find(clusters==3);
	newPos=position;
	newWeight=weight;
	count=1;

	for ii=1:length(class1),
		idx=class1(ii);
		newPos(count,:)=position(idx,:);
		newWeight(count,:)=weight(idx,:);
		count=count+1;
	end;
		
	for ii=1:length(class2),
		idx=class2(ii);
		newPos(count,:)=position(idx,:);
		newWeight(count,:)=weight(idx,:);
		count=count+1;
	end;

	for ii=1:length(class3),
		idx=class3(ii);
		newPos(count,:)=position(idx,:);
		newWeight(count,:)=weight(idx,:);
		count=count+1;
	end;

	figure(2);
	image2=reshape(newWeight,no_of_nuerons*n,1);
	image1=reshape(image2,16,16);
	imshow(double(image1));

	figure(4);
	imagesc(image1);

	figure(1);
	for kk=1:no_of_nuerons,
		%r=floor(kk/grid_size);
		%c=rem(kk,grid_size);
		%region=grid_size*(r-1)+c;
		subplot(grid_size,grid_size,kk,'align');
		img=reshape(newWeight(kk,:),2,2)';
		imshow(double(img));
		title(int2str(clusters(kk)));		
	end;
	

	%mtest=size(testX,1);
	%correct_classification=0;
	%incorrect_classification=0;
	%for ii=1:mtest,
	%	distval=zeros(no_of_nuerons,1);
	%	for kk=1:no_of_nuerons,
	%		distval(kk)=norm(testX(ii,:)-weight(kk,:));
	%	end;
		
	%	[BMU,idx]=min(distval);
	%	predicted=clusters(idx);
	%	actual=testY(ii);
	%	if(predicted==actual),
	%		correct_classification=correct_classification+1;
	%	else,
	%		incorrect_classification=incorrect_classification+1;
	%	end;		

	%end;		

	%fprintf('noof correct classifications:%d\n',correct_classification);
	%fprintf('noof incorrect classifications:%d\n',incorrect_classification);

end

