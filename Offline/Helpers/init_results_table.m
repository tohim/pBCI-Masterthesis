function results = init_results_table(dataset_names, calib_types)

% Create rows: [Dataset (Hyper: BOOL)]
rows = {};
for i = 1:numel(dataset_names)
    for hyper = [false true]
        rows{end+1,1} = sprintf('%s (Hyper: %s)', dataset_names{i}, upper(string(hyper)));
    end
end

% Create columns: [Calibrationset (Calibration Type)]
cols = {};
for i = 1:numel(dataset_names)
    for j = 1:numel(calib_types)
        cols{end+1,1} = sprintf('%s (%s)', dataset_names{i}, calib_types{j});
    end
end

% Initialize empty table with cell content
results = cell2table(cell(numel(rows), numel(cols)), ...
    'VariableNames', matlab.lang.makeValidName(cols));
results.Properties.RowNames = rows;

end