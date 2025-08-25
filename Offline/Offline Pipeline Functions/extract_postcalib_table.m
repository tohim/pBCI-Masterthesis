function T_post_cross_accuracies = extract_postcalib_table(postcalib_file, config_tags)
% Extract all cross-dataset results from AfterCalibration file
% Updated to include 'Config' tag and align naming with PreCalib table

all_rows = {};

for i = 1:length(config_tags)
    tag = config_tags{i};  % This is the config tag: "24", "csp", etc.
    sheetname = ['CalibResults_' tag];

    try
        T = readtable(postcalib_file, 'Sheet', sheetname, 'ReadRowNames', true);
    catch
        warning('[SKIP] Could not read sheet: %s', sheetname);
        continue;
    end

    rowNames = string(T.Properties.RowNames);
    colNames = string(T.Properties.VariableNames);

    for r = 1:height(T)
        row_label = rowNames(r);

        % Extract source dataset and Hyper flag
        row_match = regexp(row_label, '([A-Za-z0-9_]+)\s*\(Hyper:\s*(TRUE|FALSE)\)', 'tokens', 'once');
        if isempty(row_match), continue; end
        source_ds = row_match{1};
        hyper_flag = row_match{2};

        for c = 1:width(T)
            col_label = colNames{c};

            % Extract target dataset and calibration type from column name
            col_match = regexp(col_label, '([A-Za-z0-9_]+)_(adapted|finetuned|finetuned_adapted)_$', 'tokens', 'once');
            if isempty(col_match), continue; end
            target_ds = col_match{1};
            calib_type = col_match{2};

            % Extract cell content and accuracy
            str = T{r, c}{1};
            if isempty(str), continue; end
            str = regexprep(str, '\s+', ' ');  % flatten whitespace

            match = regexp(str, 'Cross:\s*([\d.]+)%', 'tokens', 'once');
            if isempty(match), continue; end
            acc = str2double(match{1});

            % Refine CalibrationType for NORM/HYPER NORM using cell content
            if ismember(calib_type, {'adapted', 'finetuned_adapted'})
                if contains(str, "NoNewModel(DA)")
                    calib_type = "adapted";
                else
                    calib_type = "finetuned_adapted";
                end
            end

            % Determine model type based on calib_type and hyper_flag
            if calib_type == "adapted"
                model_type = "HYPER NORM";
                if hyper_flag == "FALSE"
                    model_type = "NORM";
                end

            elseif calib_type == "finetuned_adapted"
                model_type = "HYPER NORM";
                if hyper_flag == "FALSE"
                    model_type = "NORM";
                end

            elseif calib_type == "finetuned"
                model_type = "HYPER";
                if hyper_flag == "FALSE"
                    model_type = "STANDARD";
                end
            end

            % Finalize SOURCE, TARGET, and CONFIG
            source = source_ds + " " + tag + " " + model_type;
            target = target_ds + " " + tag + " " + model_type;
            config  = tag;

            % Store the row
            all_rows = [all_rows; {
                source, target, acc, model_type, calib_type, config
            }];
        end
    end
end

% Convert to table
if isempty(all_rows)
    T_post_cross_accuracies = table();
else
    T_post_cross_accuracies = cell2table(all_rows, ...
        'VariableNames', {'SOURCE','TARGET','ACCURACY','Model','CalibrationType','Config'});
end

% Save result
save('PostCalib_Cross_Results.mat', 'T_post_cross_accuracies');
fprintf('Saved POST calib table with %d entries → PostCalib_Cross_Results.mat\n', height(T_post_cross_accuracies));

end
