function disc = Discretize(testRow,prob)

	m=length(testRow);
	disc=zeros(m,1);
    if(prob==1),

	   for ii=1:m,
            val=testRow(ii);
            if (val>=-3 && val< -1),
        	   disc(ii)=1;
            elseif (val>=-1 && val< 1),
        	   disc(ii)=2;	
            elseif (val>=1 && val< 3),
        	   disc(ii)=3;
            elseif (val>=3 && val< 5),
        	   disc(ii)=4;
            elseif (val>=5 && val< 7),
        	   disc(ii)=5;
      	
            end;
        end;

    else,

        val1=testRow(1);
        val2=testRow(2);
        val3=testRow(3);
        if(val1>=60 && val1<65),
            disc(1)=1;
         elseif(val1>=65 && val1<70),
            disc(1)=2;
         elseif(val1>=70 && val1<75),
            disc(1)=3;
         elseif(val1>=75 && val1<80),
            disc(1)=4;
         elseif(val1>=80),
            disc(1)=5;
        end;
        
        if(val2>=50 && val2<55),
            disc(2)=1;
         elseif(val2>=55 && val2<60),
            disc(2)=2;
         elseif(val2>=60 && val2<65),
            disc(2)=3;
         elseif(val2>=65 && val2<70),
            disc(2)=4;
         elseif(val2>=70),
            disc(2)=5;
        end;        

        if(val3>=0 && val3<5),
            disc(3)=1;
         elseif(val3>=5 && val3<10),
            disc(3)=2;
         elseif(val3>=10 && val3<15),
            disc(3)=3;
         elseif(val3>=15 && val3<20),
            disc(3)=4;
         elseif(val3>=20),
            disc(3)=5;
        end;                  
    end;

end

