%% Version 1

% Load Cross_All sheet
filename = 'v1_Full_Final_Summary_ALL.xlsx';
cross_data = readtable(filename, 'Sheet', 'Cross_All');

% Clean Model and CalibrationType columns
cross_data.Model = upper(strtrim(string(cross_data.Model)));
cross_data.CalibrationType = lower(strtrim(string(cross_data.CalibrationType)));

% Define calibration strategies to loop over
calibrations = {'finetuned', 'finetuned_adapted'};

for c = 1:length(calibrations)

    calib_type = calibrations{c};

    % Filter for Post-calibration, 26wCsp config, 30% calib, and current calibration type
    filtered = cross_data( ...
        strcmp(cross_data.Stage, 'Post') & ...
        strcmp(cross_data.Config, '25wCsp') & ...
        cross_data.CalibPercent == 30 & ...
        strcmp(cross_data.CalibrationType, calib_type), :);

    % Normalize and extract model strings
    filtered.Model = upper(strtrim(string(filtered.Model)));
    model_labels = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};

    % Init results
    mean_acc = zeros(length(model_labels),1);
    std_acc = zeros(length(model_labels),1);
    group_count = zeros(length(model_labels),1);

    % Manual grouping
    for i = 1:length(model_labels)
        label = model_labels{i};
        idx = strcmp(filtered.Model, label);
        mean_acc(i) = mean(filtered.ACCURACY(idx));
        std_acc(i) = std(filtered.ACCURACY(idx));
        group_count(i) = sum(idx);
    end

    % Summary table
    model_summary = table( ...
        categorical(model_labels(:)), ...
        group_count(:), ...
        mean_acc(:), ...
        std_acc(:), ...
        'VariableNames', {'Model', 'GroupCount', 'Mean_Accuracy', 'Std_Accuracy'});
    model_summary = sortrows(model_summary, 'Mean_Accuracy', 'descend');

    % Display
    disp(['=== Cross-Data Model Performance Summary (25wCsp, 30% Calibration, ' upper(calib_type) ') ===']);
    disp(model_summary);

    % Plot
    figure;
    bar(model_summary.Model, model_summary.Mean_Accuracy); hold on;
    errorbar(model_summary.Model, model_summary.Mean_Accuracy, model_summary.Std_Accuracy, ...
        'k.', 'LineWidth', 1.5);
    ylabel('Cross Accuracy (%)');
    title(['Model Comparison (25wCsp, 30% - ' upper(calib_type) ')']);
    grid on;
end


%% Version 2 — Same Logic

filename = 'v2_Full_Final_Summary.xlsx';
cross_data = readtable(filename, 'Sheet', 'Cross_All');
calibrations = {'finetuned', 'finetuned_adapted'};

% Clean Model and CalibrationType columns
cross_data.Model = upper(strtrim(string(cross_data.Model)));
cross_data.CalibrationType = lower(strtrim(string(cross_data.CalibrationType)));

for c = 1:length(calibrations)

    calib_type = calibrations{c};

    % Filter for Post-calibration, 16wCsp config, 30% calib, and current calibration type
    filtered = cross_data( ...
        strcmp(cross_data.Stage, 'Post') & ...
        strcmp(cross_data.Config, '16wCsp') & ...
        cross_data.CalibPercent == 30 & ...
        strcmp(cross_data.CalibrationType, calib_type), :);

    filtered.Model = upper(strtrim(string(filtered.Model)));
    model_labels = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};

    mean_acc = zeros(length(model_labels),1);
    std_acc = zeros(length(model_labels),1);
    group_count = zeros(length(model_labels),1);

    for i = 1:length(model_labels)
        label = model_labels{i};
        idx = strcmp(filtered.Model, label);
        mean_acc(i) = mean(filtered.ACCURACY(idx));
        std_acc(i) = std(filtered.ACCURACY(idx));
        group_count(i) = sum(idx);
    end

    model_summary = table( ...
        categorical(model_labels(:)), ...
        group_count(:), ...
        mean_acc(:), ...
        std_acc(:), ...
        'VariableNames', {'Model', 'GroupCount', 'Mean_Accuracy', 'Std_Accuracy'});
    model_summary = sortrows(model_summary, 'Mean_Accuracy', 'descend');

    disp(['=== Cross-Data Model Performance Summary (16wCsp, 30% Calibration, ' upper(calib_type) ') ===']);
    disp(model_summary);

    figure;
    bar(model_summary.Model, model_summary.Mean_Accuracy); hold on;
    errorbar(model_summary.Model, model_summary.Mean_Accuracy, model_summary.Std_Accuracy, ...
        'k.', 'LineWidth', 1.5);
    ylabel('Cross Accuracy (%)');
    title(['Model Comparison (16wCsp, 30% - ' upper(calib_type) ')']);
    grid on;
end


