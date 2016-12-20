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
set(0, 'DefaultLineLinewidth', 1.5)

%%

vec = 6:2:18;
global lambda; global method; global alt; global CompNJplot; global ran;

method = 1; %  L1 or L0
alt = 20; % iterations between dictionary update
CompNJplot = 0; % Do not plot Convergence results
rand_num = 1 : 5;
subset_vals = zeros(6e4,length(rand_num));
filename = ['lambdatst_L',num2str(method),'_long'];
    

ed_Sprsity_N2 = zeros(length(vec),length(rand_num)); %Npar/Jpar
ed_Sprsity_N3 = ed_Sprsity_N2; %Npar/BCD (SOUP)
ed_Sprsity_ext = ed_Sprsity_N2; %SOUP-DILLO
ed_Sprsity_t = ed_Sprsity_N2; %C, first and then D update for SOUP.

%similar naming scheme for NSRE
ed_NSREN2 = ed_Sprsity_N2;
ed_NSREN3 = ed_Sprsity_N2;
ed_NSRE_ext = ed_Sprsity_N2;
ed_NSRE_t = ed_Sprsity_N2;

Norm1_Z=[];

sanity_list = []; % get rid after sanity check
% h = waitbar(0,'1','Name',filename);
for ran = rand_num
    idx=1;
    sanity_list = [sanity_list ran];
    for lambda = vec
%      waitbar((idx+((ran-1)*(lambda/2)))/length(lambda)*length(rand_num),...
%          h,sprintf('random number generator set to %d, lambda = %d',ran,lambda));
     fprintf('Checking Parameter lambda = %d for Method : L%d \n',lambda,method);
     CompNJext;
     subset_vals(:,ran)=subset;
     Norm1_Z=[Norm1_Z sum(sum(abs(ZN2(:,:,end))))];
     

     ed_Sprsity_N2(idx,rand_num) = SparsityN2(end);

     ed_Sprsity_N3(idx,rand_num) = SparsityN3(end);

     ed_Sprsity_ext(idx,rand_num) = gather(Sparsity_ext(end));

%      ed_Sprsity_t(idx,rand_num) =  gather(Sparsity_t(end));


     %similar naming scheme for NSRE
     ed_NSREN2(idx,rand_num) = NSREN2(end);
     ed_NSREN3(idx,rand_num) = NSREN3(end);
     ed_NSRE_ext(idx,rand_num) = gather(NSRE_ext(end));
%      ed_NSRE_t(idx,rand_num) = gather(NSRE_t(end));
     idx=idx+1;
    end
%     close(h);
end

%%
lambda = vec;

%{

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

%}
%%

for i = rand_num
    figure(i)
    plot(100*ed_Sprsity_N2(:,i), 100*ed_NSREN2(:,i),'-o');hold on
    plot(100*ed_Sprsity_N3(:,i), 100*ed_NSREN3(:,i),'-o');hold on
    plot(100*ed_Sprsity_ext(:,i), 100*ed_NSRE_ext(:,i),'-x');hold on
%     plot(100*ed_Sprsity_t(:,i), 100*ed_NSRE_t(:,i),'-x');hold off
    legend('ed NSREN2(Npar/Jpar)','ed NSREN3(Npar/BCD D update)','ed NSRE ext(SOUP-DILLO)','ed NSRE t(SOUP C,d1,d2,..dj)');
    xlabel('Sparsity'); ylabel('NSRE');
    title(['Sparsity vs NSRE for Method: ',num2str(method),' and rng = ',num2str(rand_num(i))])
end
%%
timestamp = datestr(datetime());
filename = ['lambdatst for rng L',num2str(method),' long alt ',num2str(alt),' ',timestamp,'.mat'];
save(filename);
mailprefs;
sendmail('anishl@umich.edu','Experiment Completed',['The results are stored in file: ',filename]);