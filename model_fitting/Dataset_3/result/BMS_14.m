results(1) = importdata('./results_M4_4.mat');
results(2) = importdata('./results_M4_9.mat');
results(3) = importdata('./results_M3_4_4.mat');
results(4) = importdata('./results_M3_4_9.mat');
results(5) = importdata('./results_M2_4_4.mat');
results(6) = importdata('./results_M2_4_9.mat');

results(7) = importdata('./results_M21_4_4.mat');
results(8) = importdata('./results_M21_4_9.mat');
results(9) = importdata('./results_M21_3_4_4.mat');
results(10) = importdata('./results_M21_3_4_9.mat');
results(11) = importdata('./results_M21_2_4_4.mat');
results(12) = importdata('./results_M21_2_4_9.mat');

results(13) = importdata('./results_WSLS.mat');
results(14) = importdata('./results_M0_0.mat');

for t = 1:14
    disp(t);
    disp((sum(results(t).aic) - sum(results(14).aic))/197);
end

bms_results = mfit_bms(results);
% figure;
% bar(bms_results.xp); colormap bone;
% set(gca,'XTickLabel',{'Model 1' 'Model 2' 'Model 3''Model 4'},'FontSize',25,'YLim',[0 1.2]);
% ylabel('Exceedance probability','FontSize',25);
% title('Bayesian model comparison','FontSize',25);

figure;
bar(bms_results.pxp); colormap bone;
set(gca,'XTickLabel',{'Model 1' 'Model 2' 'Model 3' 'Model 4' 'Model 5' 'Model 6' 'Model 7' 'Model 8' 'Model 9' 'Model 10' 'Model 11' 'Model 12' 'Model 13' 'Model 14' },'FontSize',25,'YLim',[0 1.2]);
ylabel('Protected Exceedance probability','FontSize',25);
title('Bayesian model comparison','FontSize',25);

% bms_results.bor

% [sum(results(1).aic),sum(results(2).aic),sum(results(3).aic)]
bms_results.pxp