function knearest()

	X=load('breast-cancer-data.txt');
	[m,n]=size(X);
	Y=X(:,n);
	X(:,n)=[];
	X(:,1)=[];
	n=n-1;
	%X=featureScale(X);
	X=[X Y];
	k=5;
	kvalues=[1,2,3,5,7,9,11,13,15];
	len=length(kvalues);
	set1=cvpartition(m,'kfold',k);
	avgAccuracy=zeros(len,1);
	
	for ii=1:len,
		kval=kvalues(ii);
		accuracy=zeros(k,1);

		for jj=1:k,

			ip=X(training(set1,jj),:);
			op=ip(:,n);
			ip(:,n)=[];
			tstip=X(test(set1,jj),:);
			tstop=tstip(:,n);
			tstip(:,n)=[];
			trainlen=size(ip,1);
			testlen=size(tstip,1);
			outClass=zeros(testlen,1);
			confMatrix=zeros(testlen,1);

			for kk=1:testlen,

				distval=zeros(trainlen,1);
				classes=zeros(kval,1);

				for ll=1:trainlen,
					distval(ll)=norm(tstip(kk,:)-ip(ll,:)); %getDistance(tstip(kk),ip(ll));
				end;

				for ll=1:kval,
					[~,idx]=min(distval);
					if(op(idx)==2),
						classes(ll)=2;
                    else
						classes(ll)=4;
					end;
					distval(idx)=[];	
				end;

				negY=find(classes==2);
				posY=find(classes==4);
				%length(posY)
				%length(negY)		

				if(length(posY) > length(negY)),
					outClass(kk)=4;
                else
					outClass(kk)=2;	
				end;

				predictedClass=outClass(kk);
				actualClass=tstop(kk);
				if(predictedClass==4 && actualClass==4),
					confMatrix(kk)=1;
				elseif(predictedClass==4 && actualClass==2)
					confMatrix(kk)=2;	
				elseif(predictedClass==2 && actualClass==4)
					confMatrix(kk)=3;
				elseif(predictedClass==2 && actualClass==2)
					confMatrix(kk)=4;
				end;	

			end;

			sumTP=length(find(confMatrix==1));
			sumFP=length(find(confMatrix==2));
			sumFN=length(find(confMatrix==3));
			sumTN=length(find(confMatrix==4));

			accuracy(jj)=(sumTP+sumTN)/(sumTP+sumFP+sumFN+sumTN);
			%precision=sumTP/(sumTP+sumFP);
			%recall=sumTP/(sumTP+sumFN);
			%fMeasure=2*precision*recall/(precision+recall);

		end;

		avgAccuracy(ii)=mean(accuracy)	
		

	end;


	%%%%%%  CROSS VALIDATION ENDS HERE  %%%%%%%

	[maxAccuracy,idx]=max(avgAccuracy);		
	kval=kvalues(idx);
	set1=cvpartition(m,'holdout',0.3);
	ip=X(training(set1),:);
	op=ip(:,n);
	ip(:,n)=[];
	tstip=X(test(set1),:);
	tstop=tstip(:,n);
	tstip(:,n)=[];
	trainlen=size(ip,1);
	testlen=size(tstip,1);
	scores=zeros(testlen,1);

	for kk=1:testlen,

		distval=zeros(trainlen,1);
		classes=zeros(kval,1);

		for ll=1:trainlen,
			distval(ll)=norm(tstip(kk,:)-ip(ll,:)); %getDistance(tstip(kk),ip(ll));
		end;

		for ll=1:kval,
			[~,idx]=min(distval);
			if(op(idx)==2),
				classes(ll)=2;
            else
				classes(ll)=4;
			end;
			distval(idx)=[];	
		end;

		negY=find(classes==2);
		posY=find(classes==4);
		%length(posY)
		%length(negY)		

		if(length(posY) > length(negY)),
			outClass(kk)=4;
        else
			outClass(kk)=2;	
		end;

		
		scores(kk)=length(posY)/kval;
		predictedClass=outClass(kk);
		actualClass=tstop(kk);
		if(predictedClass==4 && actualClass==4),
			confMatrix(kk)=1;
		elseif(predictedClass==4 && actualClass==2)
			confMatrix(kk)=2;	
		elseif(predictedClass==2 && actualClass==4)
			confMatrix(kk)=3;
		elseif(predictedClass==2 && actualClass==2)
			confMatrix(kk)=4;
		end;	

	end;

	sumTP=length(find(confMatrix==1));
	sumFP=length(find(confMatrix==2));
	sumFN=length(find(confMatrix==3));
	sumTN=length(find(confMatrix==4));

	finalAccuracy=(sumTP+sumTN)/(sumTP+sumFP+sumFN+sumTN);
	precision=sumTP/(sumTP+sumFP);
	recall=sumTP/(sumTP+sumFN);
	fMeasure=2*precision*recall/(precision+recall);

	fprintf('no of test datasets :%d\n',testlen);
    fprintf('no of misclassifications :%d\n',sumFP+sumFN);
	fprintf('accuracy :%.3f\n',finalAccuracy);
	fprintf('precision :%.3f\n',precision);
	fprintf('recall/sensitivity :%.3f\n',recall);
	fprintf('F-Measure :%.3f\n',fMeasure);
    fprintf('Max accuracy during cross validation:%0.3f\n',maxAccuracy);
    fprintf('optimum K value:%d\n',kval);

    %%%%%%   ROC CURVE  %%%%%%%
	labels=tstop;
    [P,Q]=perfcurve(labels,scores,4);
    plot(P,Q,'LineWidth',5);

end

