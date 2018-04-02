function [jcw,old_cluster_assigned] = kmedioid(A,kval,metric)

	[m,n]=size(A);
	k=kval;
	maxiter=30;
	%metric=2;
	%jcw=zeros(maxiter,(k*n)+1);
	jcw=0;
	mu_old=zeros(1,n);


	%% Assign points as centroids randomly %%
	mu=zeros(k,n);
	for ii=1:k,
		randno=randi([1,m]);
		mu(ii,:)=A(randno,:);
	end;
	%mu
	%mu(1,:)=[5 8];
	%mu(2,:)=[7 5];	

	%% calculate the first jcw value accroding to this assignment %%
	%jcwtemp=0;
	old_cluster_assigned=zeros(m,1);
	for jj=1:m,

		distval=zeros(k,1);
		for kk=1:k,
			distval(kk)=norm(A(jj,:)-mu(kk,:),metric);
		end;
		[mindist,idx]=min(distval);
		old_cluster_assigned(jj)=idx;
		jcw=jcw+mindist;	

	end;	

	%% save the centroids and the associated cost %%
	%jcw=jcwtemp;%jcw(1,1)=jcwtemp;
	%ind=2;
	%for kk=1:k,
	%	jcw(1,ind:ind+1)=mu(kk,:);
	%	ind=ind+2;
	%end;	


	%% start the iterative updates %%
	for ii=2:maxiter,

		%% finding the new centroid to swap with older ones %%
		swapping_centroid=rem(ii,k)+1;
		cluster_idx=find(old_cluster_assigned==swapping_centroid);
		len=length(cluster_idx);
		if(len~=0),
			randidx=randi([1,len]);
			randidx=cluster_idx(randidx);
			mu_old=mu(swapping_centroid,:);
			mu(swapping_centroid,:)=A(randidx,:);
		end;	



		%% find new clusters %%
		cluster_assigned=zeros(m,1);
		jcwnew=0;
		for jj=1:m,

			distval=zeros(k,1);
			for kk=1:k,
				distval(kk)=norm(A(jj,:)-mu(kk,:),metric);
			end;
			[mindist,idx]=min(distval);
			cluster_assigned(jj)=idx;
			jcwnew=jcwnew+mindist;	

		end;	

		%%  if the new jcw values not better than old one, retain the previous centroids %%
		if(jcwnew>jcw),
			mu(swapping_centroid,:)=mu_old;
		else,
			jcw=jcwnew;
			old_cluster_assigned=cluster_assigned;
		end;

		%mu

	end;	


	%% calculate the Davis Bouldin index %%%
	sc=zeros(k,1);
	for ii=1:k,
		cluster_idx=find(old_cluster_assigned==ii);
		sc_dist=0;
		len=length(cluster_idx);
		for jj=1:len,
			idx=cluster_idx(jj);
			sc_dist=sc_dist+norm(A(idx,:)-mu(ii,:),metric);
		end;
		sc(ii)=sc_dist;	
	end;
	%sc
	d_max=0;

	for ii=1:k,
		tempdist=zeros(k-1,1);
		count=1;
		for jj=1:k,
			if(jj~=ii),
				d_centroid=norm(mu(ii,:)-mu(jj,:),metric);
				tempdist(count)=(sc(ii)+sc(jj))/d_centroid;
				count=count+1;
			end;				
		end;

		d_max=d_max+max(tempdist);	
	end;

	db_idx=d_max/k;	
	fprintf('k value:%d\n',k);
	fprintf('DB INDEX :%f\n',db_idx);

end

