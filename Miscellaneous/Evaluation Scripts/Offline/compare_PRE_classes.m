% Compare Class Wise Metrics

% Select version
version_tag = 'v1'; % change to 'v2' if needed

base_path = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
                     'Code', 'Matlab', 'Data', 'AutoPipeline', version_tag, '1000samples_30Pct_Calib');

filename = fullfile(base_path, 'PreCalib_1000_Samples_Within_Cross_Results.xlsx');

% Define dataset suffixes based on sheet names 
datasets = {'STEW', 'HEATCHAIR', 'EasyDiff', 'EasyMedDiff'};  % Sheet suffixes

% Collect class-wise data from all sheets
all_class_data = [];

fprintf('\nEvaluation of CLASS WISE Metrics PRE Calibration, Version %s: \n', version_tag);
fprintf('\n');

for i = 1:length(datasets)
    sheetname = ['Classes_' datasets{i}];
    
    try
        T = readtable(filename, 'Sheet', sheetname);
    catch
        fprintf('⚠️  Sheet %s not found in %s\n\n', sheetname, filename);
        continue;
    end

    % Normalize fields
    T.Model = upper(strtrim(string(T.Model)));
    T.Class = strtrim(string(T.Class));
    T.Source = strtrim(string(T.Source));
    T.Target = strtrim(string(T.Target));

    % Add Config based on whether Source or Target contains "26wCsp"
    T.Config = repmat("unknown", height(T), 1);
    T.Config(contains(T.Source, '25wCsp') | contains(T.Target, '25wCsp')) = "25wCsp";

    % Exclude MATB_easy_diff <-> MATB_easy_meddiff combinations
    invalid_pairs = (contains(T.Source, 'MATB_easy_diff') & contains(T.Target, 'MATB_easy_meddiff')) | ...
                    (contains(T.Source, 'MATB_easy_meddiff') & contains(T.Target, 'MATB_easy_diff'));
    T = T(~invalid_pairs, :);

    % Filter: HYPER model, finetuned, and 26wCsp config
    isValid = strcmp(T.Model, 'HYPER') & strcmp(T.Config, '25wCsp') & strcmp(T.Target, 'Within');
    T = T(isValid, :);

    % Append to global table
    all_class_data = [all_class_data; T];
end

% === Print Per-Source Dataset Class-Wise Summary ===
sources = unique(all_class_data.Source);

for s = 1:length(sources)
    src = sources{s};
    subset = all_class_data(strcmp(all_class_data.Source, src), :);
    
    fprintf('=== Class-wise Metrics by Source: %s ===\n', src);
    for class = ["Low", "High"]
        rows = strcmpi(subset.Class, class);
        if any(rows)
            prec = mean(subset.Precision(rows), 'omitnan');
            rec  = mean(subset.Recall(rows), 'omitnan');
            f1   = mean(subset.F1(rows), 'omitnan');
            fprintf('Class %-4s | Precision = %.3f | Recall = %.3f | F1 = %.3f\n', ...
                class, prec, rec, f1);
        else
            fprintf('Class %-4s | No data available.\n', class);
        end
    end
    fprintf('\n');
end


%% Version 2

% Compare Class Wise Metrics

% Select version
version_tag = 'v2';

base_path = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
                     'Code', 'Matlab', 'Data', 'AutoPipeline', version_tag, '1000samples_30Pct_Calib');

filename = fullfile(base_path, 'PreCalib_1000_Samples_Within_Cross_Results.xlsx');

% Define dataset suffixes based on sheet names 
datasets = {'STEW', 'HEATCHAIR', 'EasyDiff', 'EasyMedDiff'};  % Sheet suffixes

% Collect class-wise data from all sheets
all_class_data = [];

fprintf('\nEvaluation of CLASS WISE Metrics PRE Calibration, Version %s: \n', version_tag);
fprintf('\n');

for i = 1:length(datasets)
    sheetname = ['Classes_' datasets{i}];
    
    try
        T = readtable(filename, 'Sheet', sheetname);
    catch
        fprintf('⚠️  Sheet %s not found in %s\n\n', sheetname, filename);
        continue;
    end

    % Normalize fields
    T.Model = upper(strtrim(string(T.Model)));
    T.Class = strtrim(string(T.Class));
    T.Source = strtrim(string(T.Source));
    T.Target = strtrim(string(T.Target));

    % Add Config based on whether Source or Target contains "26wCsp"
    T.Config = repmat("unknown", height(T), 1);
    T.Config(contains(T.Source, '16wCsp') | contains(T.Target, '16wCsp')) = "16wCsp";

    % Exclude MATB_easy_diff <-> MATB_easy_meddiff combinations
    invalid_pairs = (contains(T.Source, 'MATB_easy_diff') & contains(T.Target, 'MATB_easy_meddiff')) | ...
                    (contains(T.Source, 'MATB_easy_meddiff') & contains(T.Target, 'MATB_easy_diff'));
    T = T(~invalid_pairs, :);

    % Filter: HYPER model, finetuned, and 26wCsp config
    isValid = strcmp(T.Model, 'HYPER') & strcmp(T.Config, '16wCsp') & strcmp(T.Target, 'Within');
    T = T(isValid, :);

    % Append to global table
    all_class_data = [all_class_data; T];
end

% === Print Per-Source Dataset Class-Wise Summary ===
sources = unique(all_class_data.Source);

for s = 1:length(sources)
    src = sources{s};
    subset = all_class_data(strcmp(all_class_data.Source, src), :);
    
    fprintf('=== Class-wise Metrics by Source: %s ===\n', src);
    for class = ["Low", "High"]
        rows = strcmpi(subset.Class, class);
        if any(rows)
            prec = mean(subset.Precision(rows), 'omitnan');
            rec  = mean(subset.Recall(rows), 'omitnan');
            f1   = mean(subset.F1(rows), 'omitnan');
            fprintf('Class %-4s | Precision = %.3f | Recall = %.3f | F1 = %.3f\n', ...
                class, prec, rec, f1);
        else
            fprintf('Class %-4s | No data available.\n', class);
        end
    end
    fprintf('\n');
end

