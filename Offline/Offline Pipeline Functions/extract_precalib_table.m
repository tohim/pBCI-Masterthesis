function T_pre_cross_accuracies = extract_precalib_table(precalib_file)
% Extract cross-dataset results from PreCalibration file and flag by CalibrationType
% Duplicates normalized models to match both 'adapted' and 'finetuned_adapted'

% Read full table
T = readtable(precalib_file, 'Sheet', 'Total');

% Keep only rows where TARGET does NOT contain "Within"
mask = ~contains(T.TARGET, 'Within', 'IgnoreCase', true);
T = T(mask, :);

% Add CalibrationType and Config column as string types
T.CalibrationType = strings(height(T), 1);
T.Config = strings(height(T), 1);

% Initialize output table
out_rows = [];

for i = 1:height(T)
    row = T(i, :);
    
    % Extract dataset, config, model from SOURCE
    parts = strsplit(strtrim(string(row.SOURCE)));
    if numel(parts) >= 3
        dataset = parts(1);
        config = parts(2);
        model = strjoin(parts(3:end), ' ');
    else
        dataset = "UNKNOWN"; config = "UNKNOWN"; model = "UNKNOWN";
    end
    model = string(model); config = string(config);
    
    row.Config = config;

    switch model
        case {'STANDARD', 'HYPER'}
            row.Model = model;
            row.CalibrationType = "finetuned";
            row.SOURCE = dataset + " " + config + " " + model;
            row.TARGET = normalize_target(row.TARGET, config, model);
            out_rows = [out_rows; row];

        case {'NORM', 'HYPER NORM'}
            for calib_type = ["adapted", "finetuned_adapted"]
                new_row = row;
                new_row.Model = model;
                new_row.CalibrationType = calib_type;
                new_row.Config = config;
                new_row.SOURCE = dataset + " " + config + " " + model;
                new_row.TARGET = normalize_target(new_row.TARGET, config, model);
                out_rows = [out_rows; new_row];
            end

        otherwise
            row.Model = model;
            row.CalibrationType = "unknown";
            out_rows = [out_rows; row];
    end
end

T_pre_cross_accuracies = out_rows;

% Save result
save('PreCalib_Cross_Results.mat', 'T_pre_cross_accuracies');
fprintf('Saved %d pre-calib entries → PreCalib_Cross_Results.mat\n', height(T_pre_cross_accuracies));
end

% --- Helper function to normalize TARGET format ---
function tgt = normalize_target(raw_target, config, model)
    tgt = string(raw_target);
    parts = strsplit(strtrim(tgt));
    if numel(parts) >= 3
        target_ds = parts(1);
        tgt = target_ds + " " + config + " " + model;
    end
end
