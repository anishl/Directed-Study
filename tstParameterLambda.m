clc;

clear ;
close all;
%%
set(0, 'DefaultAxesLineWidth', 2.0)
set(0, 'DefaultTextFontSize', 18)
set(0, 'DefaultTextFontWeight', 'normal')
set(0, 'DefaultAxesFontSize', 18)
%set(0, 'DefaultAxesFontWeight', 'bold')
set(0, 'DefaultAxesFontWeight', 'normal')
set(0, 'DefaultLineMarkerSize', 10)

%%

global lambda; global method; global alt;

method = 0; %  L1 or L0
alt = 20; % iterations between dictionary update


ed_Sprsity_N2 = zeros(size(2:2:24)); %Npar/Jpar
ed_Sprsity_N3 = ed_Sprsity_N2; %Npar/BCD (SOUP)
ed_Sprsity_ext = ed_Sprsity_N2; %SOUP-DILLO
ed_Sprsity_t = ed_Sprsity_N2; %C, first and then D update for SOUP.

%similar naming scheme for NSRE
ed_NSREN2 = ed_Sprsity_N2;
ed_NSREN3 = ed_Sprsity_N2;
ed_NSRE_ext = ed_Sprsity_N2;
ed_NSRE_t = ed_Sprsity_N2;

idx=1;
for lambda = 2 : 2 : 24
 CompNJext;
 fprintf('Checking Parameter lambda = %d for Method : L%d',lambda,method);
 
 ed_Sprsity_N2(idx) = SparsityN2(end);

 ed_Sprsity_N3(idx) = SparsityN3(end);

 ed_Sprsity_ext(idx) = gather(Sparsity_ext(end));

 ed_Sprsity_t(idx) =  gather(Sparsity_t(end));


 %similar naming scheme for NSRE
 ed_NSREN2(idx) = NSREN2(end);
 ed_NSREN3(idx) = NSREN3(end);
 ed_NSRE_ext(idx) = gather(NSRE_ext(end));
 ed_NSRE_t(idx) = gather(NSRE_t(end));
 idx=idx+1;
end
%%
lambda = 2:2:24;
figure(1)
plot(lambda, 100*ed_Sprsity_N2,'-o');hold on
plot(lambda, 100*ed_Sprsity_N3,'-o');hold on
plot(lambda, 100*ed_Sprsity_ext,'-x');hold on
plot(lambda, 100*ed_Sprsity_t,'-x');hold off
legend('ed Sparsity N2(Npar/Jpar)','ed Sparsity N3(Npar/BCD D update)','ed Sparsity ext(SOUP-DILLO)','ed Sparsity t(SOUP C,d1,d2,..dj)');
xlabel('lambda'); ylabel('Sparsity');
title(['Sparsity vs Lambda for Method: ',method])
figure(2)
plot(lambda, 100*ed_NSREN2,'-o');hold on
plot(lambda, 100*ed_NSREN3,'-o');hold on
plot(lambda, 100*ed_NSRE_ext,'-x');hold on
plot(lambda, 100*ed_NSRE_t,'-x');hold off
legend('ed NSREN2(Npar/Jpar)','ed NSREN3(Npar/BCD D update)','ed NSRE ext(SOUP-DILLO)','ed NSRE t(SOUP C,d1,d2,..dj)');
xlabel('lambda'); ylabel('NSRE');
title(['Sparsity vs Lambda for Method: ',method])
%%
figure(3)
plot(100*ed_Sprsity_N2, 100*ed_NSREN2,'-o');hold on
plot(100*ed_Sprsity_N3, 100*ed_NSREN3,'-o');hold on
plot(100*ed_Sprsity_ext, 100*ed_NSRE_ext,'-x');hold on
plot(100*ed_Sprsity_t, 100*ed_NSRE_t,'-x');hold off
legend('ed NSREN2(Npar/Jpar)','ed NSREN3(Npar/BCD D update)','ed NSRE ext(SOUP-DILLO)','ed NSRE t(SOUP C,d1,d2,..dj)');
xlabel('Sparsity'); ylabel('NSRE');
title(['Sparsity vs NSRE for Method: ',method])
%%
timestamp = datestr(datetime());
filename = ['lambdatst_L',num2str(method),'_long_alt',num2str(alt),' ',timestamp,'.mat'];
save(filename);
mailprefs;
sendmail('anishl@umich.edu','Experiment Completed',['The results are stored in file: ',filename]);