function sort_class_metrics_sheet(filename, sheetname)
    % Load the full table from the given sheet
    try
        T = readtable(filename, 'Sheet', sheetname);
    catch
        warning('[SKIP] Failed to read sheet: %s', sheetname);
        return;
    end

    % Check CalibrationType column
    calib_order = {'adapted', 'finetuned', 'finetuned_adapted'};
    if ismember('CalibrationType', T.Properties.VariableNames)
        T.CalibrationType = categorical(T.CalibrationType, calib_order, 'Ordinal', true);
    else
        warning('[SKIP] No CalibrationType in sheet: %s', sheetname);
        return;
    end

    % Optional: prioritize "CalibrationSource (WITHIN)" in Target column
    if ismember('Target', T.Properties.VariableNames)
        T.Target = string(T.Target); % Ensure it's a string array
        target_order = ["CalibrationSource (WITHIN)", ...
                        sort(setdiff(unique(T.Target), "CalibrationSource (WITHIN)"))'];
        T.Target = categorical(T.Target, target_order, 'Ordinal', true);
    end

    % Sort by both CalibrationType and Target
    T = sortrows(T, {'CalibrationType', 'Target'});

    % Save the sorted table
    writetable(T, filename, 'Sheet', sheetname);
    fprintf('[DONE] Sorted %s by CalibrationType + Target\n', sheetname);
end
