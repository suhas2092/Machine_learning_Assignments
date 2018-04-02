function [disc,classes] = AttributeDiscretizer(X,prob)

    % this function is hardcoded since the range of values that an
    % attribute takes is varying a lot for different attributes.
    [m n]=size(X);
    disc=zeros(m,n);
    classes=5;
    if(prob==1),

        for ii=1:m,
    	     for jj=1:2,
              val=X(ii,jj);
        	    if (val>=-3 && val< -1),
        		     disc(ii,jj)=1;
        	    elseif (val>=-1 && val< 1),
        		     disc(ii,jj)=2;	
        	   elseif (val>=1 && val< 3),
        		     disc(ii,jj)=3;
        	   elseif (val>=3 && val< 5),
        		     disc(ii,jj)=4;
        	   elseif (val>=5),
        		     disc(ii,jj)=5;
        	   end;	
           end;
        end;    
    

    else,
     for ii=1:m,
        val1=X(ii,1);
        val2=X(ii,2);
        val3=X(ii,3);
        if(val1>=30 && val1<40),
            disc(ii,1)=1;
         elseif(val1>=40 && val1<50),
            disc(ii,1)=2;
         elseif(val1>=50 && val1<60),
            disc(ii,1)=3;
         elseif(val1>=60 && val1<70),
            disc(ii,1)=4;
         elseif(val1>=70),
            disc(ii,1)=5;
        end;
        
        if(val2>=50 && val2<55),
            disc(ii,2)=1;
         elseif(val2>=55 && val2<60),
            disc(ii,2)=2;
         elseif(val2>=60 && val2<65),
            disc(ii,2)=3;
         elseif(val2>=65 && val2<70),
            disc(ii,2)=4;
         elseif(val2>=70),
            disc(ii,2)=5;
        end;        

        if(val3>=0 && val3<5),
            disc(ii,3)=1;
         elseif(val3>=5 && val3<10),
            disc(ii,3)=2;
         elseif(val3>=10 && val3<15),
            disc(ii,3)=3;
         elseif(val3>=15 && val3<20),
            disc(ii,3)=4;
         elseif(val3>=20),
            disc(ii,3)=5;
        end;          
      end;
    end;  
end
               