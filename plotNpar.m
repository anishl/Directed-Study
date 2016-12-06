%% Plot results
figure(3);

% Fig. 3 (a): Objective function
subplot(1,4,1); plot(1:N+1,ObjFunc,'r'); xlim([1 N+1]);
xlabel('Iteration Number'); ylabel('Objective Function');

% Fig. 3 (b): NSRE (percentage) and sparsity factor of C (as a percentage)
subplot(1,4,2); [yyAxes,yySparsity,yyNSRE] = plotyy(1:N+1,100*Sparsity,1:N+1,100*NSRE);
xlim([1 N+1]); yySparsity.LineStyle = '-'; yyNSRE.LineStyle = '--';
xlabel('Iteration Number'); ylabel(yyAxes(1),'Sparsity (%)'); ylabel(yyAxes(2),'NSRE (%)');
legend('Sparsity (%)','NSRE (%)');

% Fig. 3 (c): Changes between successive D iterates (||Dt - Dt-1||_F)
subplot(1,4,3); semilogy(1:N+1,Dchange,'r-'); xlim([1 N+1]);
xlabel('Iteration Number'); ylabel('||D^t - D^{t-1}||_F');

% Fig. 3 (d): Normalized changes between successive C iterates (||Ct - Ct-1||_F/||Y||_F)
subplot(1,4,4); semilogy(1:N+1,Cchange,'r-'); xlim([1 N+1]);
xlabel('Iteration Number'); ylabel('||C^t - C^{t-1}||_F / ||Y||_F');
