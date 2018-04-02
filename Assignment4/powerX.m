function X = powerX(ip,pow)

	y=ip(:,3);
	x=ip(:,2);
	if(pow==2),
		X=[ ip(:,1) x x.^2 y ];
	else if(pow==3),
		X=[ ip(:,1) x x.^2 x.^3 y ];
	else,
		X=[ ip(:,1) x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 y];	
	
	end;

end

