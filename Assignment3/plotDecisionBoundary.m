function plotDecisionBoundary(theta, X, y)

	pos=find(y==1); 
	neg=find(y==0);
	%format long;
	figure(5);
	plot(X(pos, 2), X(pos,3), '+','Color','r');
	hold on;
	plot(X(neg, 2), X(neg,3), 'o');
	hold on;

	plot_x1 = [min(X(:,2))-2,  max(X(:,2))+2];
	plot_x2 = [min(X(:,3))-2,  max(X(:,3))+2];

	y1= -(theta(1)+theta(2)*plot_x1(1))/theta(3);
	y2= -(theta(1)+theta(2)*plot_x1(2))/theta(3);

	%y1= -(theta(1)+theta(2)*plot_x1(1)+theta(3)*plot_x2(1))/theta(4);
	%y2= -(theta(1)+theta(2)*plot_x1(2)+theta(3)*plot_x2(2))/theta(4);
	plot_y=[y1 y2];
	%plot_y
	%plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
	%plot_y=[7.0720  -10.8853];
	plot(plot_x1,plot_y,'Color','k');
	legend('Positive', 'Negative', 'DB');

	hold off;

end