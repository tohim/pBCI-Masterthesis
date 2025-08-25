function export_log(params, acc1, acc2)

timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
log_filename = ['log_' upper(params.dataset) '.txt'];
fid = fopen(log_filename, 'a'); % Append mode

fprintf(fid, '\n[%s]\n', timestamp);
fprintf(fid, 'Calibration Set:           %s\n', params.calibrationset);
fprintf(fid, 'Calibration Type:          %s\n', params.calibration);
fprintf(fid, 'Model Type:                %s\n', params.modeltype);
fprintf(fid, 'Hyperparameter Tuned:      %s\n', string(params.hyper));

if ~isempty(acc1)
    fprintf(fid, 'Validation Accuracy (Src): %.2f%%\n', acc1);
end
if ~isempty(acc2)
    fprintf(fid, 'Validation Accuracy (Cross): %.2f%%\n', acc2);
end

fprintf(fid, '-------------------------------------------------\n');
fclose(fid);
fprintf('[INFO] Appended results to log file: %s\n', log_filename);
end