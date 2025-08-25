function move_delta_into_stats(params)
% MOVE_DELTA_INTO_STATS - Appends the Delta sheet from the Calibration_Delta_Results file
% into the corresponding AfterCalib Stats file, and deletes the Delta file.

% Auto-generate stats filename from params
stats_file = sprintf('AfterCalib_%dTotal_%dCalib_Stats.xlsx', ...
    params.total_samples, params.calib_samples);

% Fixed Delta file name
delta_file = 'Calibration_Delta_Results.xlsx';

% Try to read and copy the Delta Sheets to the STATS file - afterwards
% delete the Delta File.
try
    % Read Delta sheet
    T_delta = readtable(delta_file, 'Sheet', 'Delta');
    writetable(T_delta, stats_file, 'Sheet', 'Delta');
    fprintf('Delta sheet successfully copied to: %s (Sheet: Delta)\n', stats_file);

    % Read Wilcoxon Stats
    try
        T_stats = readtable(delta_file, 'Sheet', 'Wilcoxon_DeltaStats');
        writetable(T_stats, stats_file, 'Sheet', 'Wilcoxon_DeltaStats');
        fprintf('Wilcoxon stats copied to: %s (Sheet: Wilcoxon_DeltaStats)\n', stats_file);
    catch
        warning('No Wilcoxon stats found in: %s', delta_file);
    end
    
    % Read Delta Summary
    try
        T_summary = readtable(delta_file, 'Sheet', 'Delta_Summary');
        writetable(T_summary, stats_file, 'Sheet', 'Delta_Summary');
        fprintf('Delta summary copied to: %s (Sheet: Delta_Summary)\n', stats_file);
    catch
        warning('Delta summary not found in: %s', delta_file);
    end

    % Delete original delta file
    delete(delta_file);
    fprintf('Deleted original delta file: %s\n', delta_file);

catch ME
    warning(E.identifier, 'Failed to move Delta sheets: %s', ME.message);
end

end
