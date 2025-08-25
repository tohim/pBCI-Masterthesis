function results = write_to_results_table(results, params, acc1, acc2, calib_info)

% Row label: source dataset + hyper info
row_label = sprintf('%s (Hyper: %s)', params.dataset, upper(string(params.hyper)));

% Column label: target dataset + calibration type
col_label = sprintf('%s (%s)', params.calibrationset, params.calibration);
col_label = matlab.lang.makeValidName(col_label);  % ensure MATLAB valid column name

% If Only Domain Adaptation and no new model trained:
if isnan(acc1)
    % Prepare entry: Source = No Now Model was trained as only Domain Adaptation was tested & Cross Acc + Sample Info
    acc_text = sprintf('Source: NoNewModel(DA) | Cross: %.2f%%\n | Samples: %d/%d (%.2f%%)', ...
        acc2, calib_info.samples, round(calib_info.source_total), calib_info.ratio);
else
    % Prepare entry: Source & Cross Acc + Sample Info (Ratio Total Samples vs Calibration Samples)
    acc_text = sprintf('Source: %.2f%% | Cross: %.2f%%\n | Samples: %d/%d (%.2f%%)', ...
        acc1, acc2, calib_info.samples, round(calib_info.source_total), calib_info.ratio);
end

% Write entry to table
results{row_label, col_label} = {acc_text};

end