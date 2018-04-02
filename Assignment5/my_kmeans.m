function [jcw,old_cluster_assigned] = my_kmeans(A,k,metric)

	[m,n]=size(A);
	maxiter=50;
	%metric=2;
	%jcw=zeros(maxiter,(k*n)+1);
	jcw=0;
	mu_old=zeros(1,n);
	epsilon=1e-5;

	%% Assign points as centroids randomly %%
	mu=zeros(k,n);
	for ii=1:k,
		randno=randi([1,m]);
		mu(ii,:)=A(randno,:);
	end;


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


	%% start the iterative updates %%
	for ii=2:maxiter,

		%% finding the new centroid by taking the mean %%
		for jj=1:k,			
			cluster_idx=find(old_cluster_assigned==jj);
			if(length(cluster_idx)>1),
				mu(jj,:)=zeros(1,n);
				for kk=1:length(cluster_idx),
					idx=cluster_idx(kk);
					mu(jj,:)=mu(jj,:)+A(idx,:);
				end;
				mu(jj,:)=mu(jj,:)/length(cluster_idx);
			end;	

		end;	


		%% find new clusters %%
		old_cluster_assigned=zeros(m,1);
		jcwnew=0;
		for jj=1:m,

			distval=zeros(k,1);
			for kk=1:k,
				distval(kk)=norm(A(jj,:)-mu(kk,:),metric);
			end;
			[mindist,idx]=min(distval);
			old_cluster_assigned(jj)=idx;
			jcwnew=jcwnew+mindist;	

		end;	

		if(norm(jcw-jcwnew)<epsilon),
			break;
		else,	
			jcw=jcwnew;
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

