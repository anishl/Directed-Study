Sct_4_1_ConvergenceExperiment_sanityscript
obj=ObjFunc(end);
sprs=100*Sparsity(end);
K=1000;
[D,C,ObjFunc,Sparsity,NSRE,Dchange,Cchange] = SOUP_DILLO_test(Y,J,lambda,K,L,D,C);
objn=ObjFunc(end);
sprsn=100*Sparsity(end);

figure(4);

% Fig. 4 (a): Objective function
subplot(1,4,1); plot(1:K,ObjFunc,'r'); xlim([1 K]);
xlabel('Iteration Number'); ylabel('Objective Function');

% Fig. 4 (b): NSRE (percentage) and sparsity factor of C (as a percentage)
subplot(1,4,2); [yyAxes,yySparsity,yyNSRE] = plotyy(1:K,100*Sparsity,1:K,100*NSRE);
xlim([1 K]); yySparsity.LineStyle = '-'; yyNSRE.LineStyle = '--';
xlabel('Iteration Number'); ylabel(yyAxes(1),'Sparsity (%)'); ylabel(yyAxes(2),'NSRE (%)');
legend('Sparsity (%)','NSRE (%)');

% Fig. 4 (c): Changes between successive D iterates (||Dt - Dt-1||_F)
subplot(1,4,3); semilogy(1:K,Dchange,'r-'); xlim([1 K]);
xlabel('Iteration Number'); ylabel('||D^t - D^{t-1}||_F');

% Fig. 4 (d): Normalized changes between successive C iterates (||Ct - Ct-1||_F/||Y||_F)
subplot(1,4,4); semilogy(1:K,Cchange,'r-'); xlim([1 K]);
xlabel('Iteration Number'); ylabel('||C^t - C^{t-1}||_F / ||Y||_F');
