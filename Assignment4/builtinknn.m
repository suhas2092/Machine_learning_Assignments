X=load('breast-cancer-data.txt');
[m,n]=size(X);
X(:,1)=[];
n=n-1;
set1=cvpartition(m,'holdout',0.3);
ip=X(training(set1),:);
op=ip(:,n);
tstip=X(test(set1),:);
tstop=tstip(:,n);
tstip(:,n)=[];
ip(:,n)=[];
len=size(tstip,1);
Class=knnclassify(tstip, ip, op, 1);

confMatrix=zeros(len,1);

for kk=1:len,
    predictedClass=Class(kk);
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

fprintf('no of test datasets :209\n');
fprintf('no of misclassifications :%d\n',sumFP+sumFN);
fprintf('accuracy :%.3f\n',finalAccuracy);
fprintf('precision :%.3f\n',precision);
fprintf('recall/sensitivity :%.3f\n',recall);
fprintf('F-Measure :%.3f\n',fMeasure);


