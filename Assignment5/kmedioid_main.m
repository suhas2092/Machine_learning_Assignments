function kmedioid_main(A,metric)
	clc;
	[m,n]=size(A);
	jcw_store=zeros(m-1,1);
	count=1;
	for ii=2:m,
		[jcw,~]=kmedioid(A,ii,metric);
		jcw_store(count)=jcw;
		count=count+1;
	end;

	%jcw_store
	figure(1);
	plot(2:m,jcw_store);
	title('plot of jcw');
	xlabel('k value');
	ylabel('jcw');	


	figure(2);
	[~,clusters]=kmedioid(A,3,metric);

	clusteridx=find(clusters==1);
	for ii=1:length(clusteridx),
		ind=clusteridx(ii);
		plot(A(ind,1),A(ind,2),'o','markers',12);
		xlim([-1,15]);
		ylim([-1,15]);
		hold on;
	end;

	clusteridx=find(clusters==2);
	for ii=1:length(clusteridx),
		ind=clusteridx(ii);
		plot(A(ind,1),A(ind,2),'r*','markers',12);
	end;

	clusteridx=find(clusters==3);
	for ii=1:length(clusteridx),
		ind=clusteridx(ii);
		plot(A(ind,1),A(ind,2),'r+','markers',12);
	end;


	%clusteridx=find(clusters==4);
	%for ii=1:length(clusteridx),
	%	ind=clusteridx(ii);
	%	plot(A(ind,1),A(ind,2),'go','markers',12);
	%end;


	%clusters

end

