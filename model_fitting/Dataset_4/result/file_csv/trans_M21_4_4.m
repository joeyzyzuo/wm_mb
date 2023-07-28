for j = 1:1108
    C{j}=num2str(j);
end
% C{1}=num2str(1);
results.id=C;
% results.id=subdata.id;
t = array2table(results.x, 'RowNames', [results.id], 'VariableNames', strrep({results.param.name}, ' ', '_'));
t.Properties.DimensionNames(1) = {'id'};
writetable(t, ['./file_csv/params_M21_4_4.csv'], 'WriteRowNames',true,'Delimiter',';')
