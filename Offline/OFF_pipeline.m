%% Offline Processing Pipeline for pBCI Classification

% Automated Offline Data Loading, Preprocessing, Feature
% Extraction and Model Training and Evaluation (Source & Target pre Calibration)

% Automated Calibration Phase including Adapted, Finetuned and
% Finetuned-Adapted Approaches together with new Model Training and
% Evaluation (Source & Target after Calibration)


%% File Naming Convention

% Total Samples Run = 1000 | 2000 | 3000 | 4000
% Hyperparameter Tuned = hyper (includes best_C, best_kernel and W_csp)
% Amount of Features: 14 / 24/ 25 | csp_4/ csp_6 (CSP with 4/6 filters / 2/3perClass) | 14wCsp / 24wCsp / 25wCsp (Standard 14/24/25 Features + CSP Features)
% Epoch Time: '2sec'; '4sec'; 
% Raw Data: 'raw'
% Processed Data: 'processed'
% 'proc3wRef'  -> Processed with DC Offset Removal, 2 to 20 Hz Butterworth 2nd order, MAD Artifact Removal, Average Referenced
% 'proc3noRef' -> Processed with DC Offset Removal, 2 to 20 Hz Butterworth 2nd order, MAD Artifact Removal, NO Average Referenced
% 'proc4' -> 2 to 20 Hz Butterworth 2nd order, MAD Artifact Removal, Average Referenced
% 'proc5' -> 2 to 20 Hz Butterworth 2nd oder, EXTENDED MAD Artifact Removal (!FINAL VERSION!)
% Dataset Name: 'STEW', 'HEATCHAIR', 'MATB' (easy_diff | easy_med_diff)
% Data Type: 'epochs' | '(sampled_)labels' | 'train/val/test_features' | 'model' | 'norm_model' (for normalized Models)
% 'finetuned' (for Models with Transfer Learning) | 'finetuned_adapted' (for Models with Transfer Learning + Domain Adaptation Data)


%% Automated Offline Pipeline
clear; close all; clc;
% -------------------------------------------------------------------------
% [CONFIGURATION]
% -------------------------------------------------------------------------
% General Options for Data Loading
opts_base = struct();
opts_base.verbose         = false;          % Control Console Printing/ Figure Creation in Automation Run
opts_base.total_samples   = 1000;           % Define Total Used Sample Size

% Define Number of Calibration Samples in % of Training Data used
opts_base.train_ratio     = 0.7;            % Ratio of Training Data Used
opts_base.calib_ratio     = 0.1;            % Ratio of Calibration Data Used
opts_base.calib_samples   = round(opts_base.calib_ratio*(opts_base.train_ratio*opts_base.total_samples));  

% Force to be even for class balance
if mod(opts_base.calib_samples, 2) ~= 0
    opts_base.calib_samples = opts_base.calib_samples + 1;  % safer to go slightly above
end

% Epoch & Feature Settings
opts_base.fs              = 128;            % Select Sampling Rate
opts_base.epochlength     = 4;              % Select specific Epoch Length (in Seconds)
opts_base.num_features    = 25;             % Select Number of Handcrafted Features
opts_base.num_csp_filters = 6;              % Select Number of CSP Filters / Features

% Set Versions as needed (for already existing versioned data loading)
versions = struct(...
    'STEW',                 'v1'   ,...
    'MATB_easy_diff',       'v1'   ,...
    'MATB_easy_meddiff',    'v1'   ,...
    'HEATCHAIR',            'v1'   ...
    );

% Set Processing Types as needed (for already existing pre-processed data loading)
proc_types = struct(...
    'STEW',                 'proc5'    , ...
    'MATB_easy_diff',       'proc5'    , ...
    'MATB_easy_meddiff',    'proc5'    , ...
    'HEATCHAIR',            'proc5'     ...
    );

% Take all combinations of datasets to loop through
datasets = {'STEW', 'MATB_easy_diff', 'MATB_easy_meddiff', 'HEATCHAIR'};

% Feature Name Mapping                                                                  % !! CHANGE IF DIFFERENT FEATURE NAMES !!

% Base 25 Handcrafted Features (v0)
handcrafted_feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', ...
    'Theta Beta Ratio', 'Engagement Index', 'Theta Frontal', 'Theta Temporal', 'Theta Parietal', ...
    'Theta Occipital', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', ...
    'Alpha Occipital', 'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
    'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', ...
    'Avg Complexity', 'Avg Entropy', 'Theta Entropy', 'Alpha Entropy'};

% Base 24 Handcrafted Features (v1)
% handcrafted_feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', ...
%     'Theta Beta Ratio', 'Alpha Beta Ratio', 'Engagement Index', 'Theta Frontal', ...
%     'Theta Parietal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', ...
%     'Alpha Occipital', 'Beta Frontal', 'Beta Temporal', 'Beta Parietal', ...
%     'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', ...
%     'Avg Complexity', 'Avg Entropy', 'Theta Entropy', 'Alpha Entropy'};

% Removed 10 Worst Features from Base (24) --> NumFeatures=14 (v2)
% handcrafted_feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Frontal', ...
%     'Theta Parietal', 'Alpha Frontal', 'Alpha_Temporal', 'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', ...
%     'Beta Parietal', 'Theta Entropy', 'Alpha Entropy'};

% For Baseline run
% csp_feature_names = {'CSP1_Low_Workload', 'CSP2_Low_Workload', ...
%                      'CSP1_High_Workload', 'CSP2_High_Workload'};

csp_feature_names = {'CSP1_Low_Workload', 'CSP2_Low_Workload', 'CSP3_Low_Workload' ...
                     'CSP1_High_Workload', 'CSP2_High_Workload', 'CSP3_High_Workload'};

combined_feature_names = [handcrafted_feature_names, csp_feature_names];

opts_base.handcrafted_feature_names = handcrafted_feature_names;
opts_base.csp_feature_names = csp_feature_names;
opts_base.combined_feature_names = combined_feature_names;


% Automatic Feature String Generation for Feature Combinations
feature_str_handcrafted = sprintf('%d', opts_base.num_features);             % e.g., '25', '24', '16', etc.
feature_str_handcrafted_wCsp = sprintf('%dwCsp', opts_base.num_features);    % e.g., '25wCsp', '24wCsp', '16wCsp', etc.

% In the Main Loop "opts.use_features" takes feature_configs and iterates through
% the 3 different feature combinations
feature_configs = {
    struct('use_features', true, 'use_csp', false, 'label', feature_str_handcrafted);
    struct('use_features', false, 'use_csp', true, 'label', 'csp');
    struct('use_features', true, 'use_csp', true, 'label', feature_str_handcrafted_wCsp);
    };

% Pre-Create Excel File with empty "Total" Sheet
output_name = sprintf('PreCalib_%d_Samples_Within_Cross_Results.xlsx', opts_base.total_samples);
opts_base.results_name = output_name;

% Initialize result table
pre_calib_results = table();
writetable(pre_calib_results, opts_base.results_name, 'Sheet', 'Total');


% -------------------------------------------------------------------------
% WITHIN DATASET 
% -------------------------------------------------------------------------
fprintf('\n[STAGE 1] Running WITHIN-Dataset Pipeline for All Datasets + Configs...\n');

for i = 1:numel(datasets)
    src = datasets{i};

    for cfg = 1:numel(feature_configs)

        % Base Opts Struct
        opts = opts_base;

        % General Config
        opts.dataset = src;
        opts.version = versions.(src);
        opts.proc = proc_types.(src);

        % Feature Config
        opts.use_features = feature_configs{cfg}.use_features;
        opts.use_csp = feature_configs{cfg}.use_csp;

        % Label for result table
        source_dataset_tag = sprintf('%s %s', src, feature_configs{cfg}.label);

        % Progress Info
        fprintf('\n=====================================\n');
        fprintf('[WITHIN] %s (%d/%d | Config %d/%d)\n', ...
            source_dataset_tag, i, numel(datasets), cfg, numel(feature_configs));
        fprintf('=====================================\n');

        % Running Within Dataset Pipeline
        [acc, acc_hyper, acc_norm, acc_hyper_norm, pct_std, pct_hyper, pct_norm, pct_hyper_norm] = run_within_dataset(opts);
        %close all;

        pre_calib_results = add_result(pre_calib_results, [source_dataset_tag ' STANDARD'],   'Within', acc);
        pre_calib_results = add_result(pre_calib_results, [source_dataset_tag ' HYPER'],      'Within', acc_hyper);
        pre_calib_results = add_result(pre_calib_results, [source_dataset_tag ' NORM'],       'Within', acc_norm);
        pre_calib_results = add_result(pre_calib_results, [source_dataset_tag ' HYPER NORM'], 'Within', acc_hyper_norm);

        % Save all 4 models
        write_class_metrics_to_excel(pct_std,        source_dataset_tag, 'Within', 'STANDARD',   opts.results_name);
        write_class_metrics_to_excel(pct_hyper,      source_dataset_tag, 'Within', 'HYPER',      opts.results_name);
        write_class_metrics_to_excel(pct_norm,       source_dataset_tag, 'Within', 'NORM',       opts.results_name);
        write_class_metrics_to_excel(pct_hyper_norm, source_dataset_tag, 'Within', 'HYPER NORM', opts.results_name);

    end
end


% -------------------------------------------------------------------------
% CROSS DATASET 
% -------------------------------------------------------------------------
fprintf('\n[STAGE 2] Running CROSS-Dataset Evaluation using Saved Features & Models...\n');

for i = 1:numel(datasets)
    src = datasets{i};

    for cfg = 1:numel(feature_configs)
        opts = opts_base;
        opts.dataset = src;
        opts.version = versions.(src);
        opts.proc = proc_types.(src);
        opts.use_features = feature_configs{cfg}.use_features;
        opts.use_csp = feature_configs{cfg}.use_csp;

        source_dataset_tag = sprintf('%s %s', src, feature_configs{cfg}.label);

        for j = 1:numel(datasets)
            if i == j, continue; end
            
            target_dataset = datasets{j};
            opts.cross_dataset = target_dataset;
            opts.cross_version = versions.(target_dataset);
            opts.cross_proc = proc_types.(target_dataset);  

            % Define matching feature config for target too
            target_dataset_tag = sprintf('%s %s', target_dataset, feature_configs{cfg}.label);

            % Progress Info
            fprintf('\n=====================================\n');
            fprintf('[CROSS] %s → %s\n', source_dataset_tag, target_dataset_tag);
            fprintf('Progress: Source %d/%d | Target %d/%d | Config %d/%d\n', ...
                i, numel(datasets), j, numel(datasets), cfg, numel(feature_configs));
            fprintf('=====================================\n');
            
            % Parameter Setup for individual Cross-Dataset Evaluation
            accs = NaN(1, 4);  % [standard, hyper, norm, hyper_norm]
            labels = {'STD', 'HYPER', 'NORM', 'HYPER NORM'};
            bools = [false, true, false, true;  % is_hyper
                false, false, true, true]; % is_norm

            % Loop through all conditions and compute Cross-Data Accuracies
            for m = 1:4
                try
                    fname = generate_model_filename(opts, bools(1, m), bools(2, m));        
                    fprintf('[INFO] Evaluating model file: %s\n', fname);
                    mdl = load_model(fname);

                    % Generate Cross-Dataset Evaluation
                    [accs(m), per_class_table] = run_cross_dataset_eval(opts, mdl);
                    %close all;
                    
                    % Write Class Metrics to excel
                    model_label = labels{m};
                    write_class_metrics_to_excel(per_class_table, source_dataset_tag, target_dataset_tag, model_label, opts.results_name);

                catch ME
                    fprintf('[WARNING] Failed: %s → %s (%s): %s\n', ...
                        opts.dataset, opts.cross_dataset, labels{m}, ME.message);
                end
            end

            pre_calib_results =  add_result(pre_calib_results, [source_dataset_tag ' STANDARD'],   [target_dataset_tag   ' STANDARD'], accs(1));
            pre_calib_results =  add_result(pre_calib_results, [source_dataset_tag ' HYPER'],      [target_dataset_tag      ' HYPER'], accs(2));
            pre_calib_results =  add_result(pre_calib_results, [source_dataset_tag ' NORM'],       [target_dataset_tag       ' NORM'], accs(3));
            pre_calib_results =  add_result(pre_calib_results, [source_dataset_tag ' HYPER NORM'], [target_dataset_tag ' HYPER NORM'] ,accs(4));
        end
    end
end


% ------------------------------------------------------------------------
% SAVE FINAL RESULTS
% -------------------------------------------------------------------------
fprintf('\n[STAGE 3] Finalizing and Saving...\n');

% Sort Results
pre_calib_results = sortrows(pre_calib_results, {'SOURCE', 'TARGET'});

% Save to Excel
writetable(pre_calib_results, opts.results_name, 'Sheet', 'Total');

% Get Top Accuracies
generate_top_accuracies_by_pair(opts.results_name, opts.total_samples, 70);


% ------------------------------------------------------------------------
% STATISTICAL EVALUATION OF THE RESULTS
% -------------------------------------------------------------------------
fprintf('\n[STAGE 4] Running Statistical Evaluation of the Results...\n');

% Compute Statistics
compute_stats(opts.results_name, opts);                                         

stats_file = sprintf('PreCalib_%dsamples_Stats.xlsx', opts.total_samples);      


% -------------------------------------------------------------------------
% FEATURE ANALYSIS
% -------------------------------------------------------------------------
fprintf('\n[STAGE 5] Running Feature Analysis per Dataset and Feature Config...\n');

for i = 1:numel(datasets)
    opts.dataset = datasets{i};
    opts.version = versions.(opts.dataset);
    opts.proc = proc_types.(opts.dataset);

    for cfg = 1:numel(feature_configs)
        opts.use_features = feature_configs{cfg}.use_features;
        opts.use_csp = feature_configs{cfg}.use_csp;
        
        % Run Feature Analysis for all features and all classes
        run_feature_analysis(opts, stats_file, feature_configs{cfg}.label);     

        % Run Class-specific Feature Analysis ("Which features are most
        % informative for detecting either LOW or HIGH MWL?")
        run_class_feature_analysis(opts, stats_file, feature_configs{cfg}.label);

    end
end

% Search for Top Features of every Config / Analysis Metric / Overall
% Custom Configs Example:
% custom_configs = {'20', '20wCsp'};
% summarize_top_features_across_configs('50samples_stats.xlsx', 10, custom_configs);

% Automated:
summarize_top_features_across_configs(opts, stats_file, 15);                   

fprintf('\n[INFO] Statistical Dataset Analysis Complete.\n');


% -------------------------------------------------------------------------
% SAVE PIPELINE METADATA
% -------------------------------------------------------------------------
fprintf('\n[STAGE 6] Saving Pipeline Metadata...\n');

% Save as .mat file
save(sprintf('pipeline_metadata_%dsamples.mat', opts.total_samples), ...
    'opts_base', 'feature_configs', 'datasets', 'versions', 'proc_types');

% Save as readable .txt file
txtfile = sprintf('pipeline_metadata_%dsamples.txt', opts.total_samples);
fid = fopen(txtfile, 'w');
fprintf(fid, '--- PIPELINE METADATA ---\n');
fprintf(fid, 'Total Samples: %d\n', opts_base.total_samples);
fprintf(fid, 'TRAIN Samples: %d\n', opts_base.total_samples * opts_base.train_ratio);
fprintf(fid, 'Calibration Samples: %d\n', opts_base.calib_samples);
fprintf(fid, 'Calibration Ratio: %.1f %%\n', opts_base.calib_ratio*100);
fprintf(fid, 'Epoch Length: %d sec\n', opts_base.epochlength);
fprintf(fid, 'Handcrafted Features: %d\n', opts_base.num_features);
fprintf(fid, 'CSP Filters: %d\n', opts_base.num_csp_filters);
fprintf(fid, 'Sampling Rate: %d Hz\n', opts_base.fs);
fprintf(fid, '\nDatasets:\n');
for d = 1:numel(datasets)
    fprintf(fid, '- %s (proc = %s, version = %s)\n', ...
        datasets{d}, proc_types.(datasets{d}), versions.(datasets{d}));
end
fprintf(fid, '\nFeature Configurations:\n');
for f = 1:numel(feature_configs)
    cfg = feature_configs{f};
    fprintf(fid, '- Label: %s | use_features = %d | use_csp = %d\n', ...
        cfg.label, cfg.use_features, cfg.use_csp);
end
fclose(fid);

fprintf('[SAVED] Metadata as .mat and .txt files.\n');


% -------------------------------------------------------------------------
% AUTOMATIC STORAGE
% -------------------------------------------------------------------------
fprintf('\n[STAGE 7] Storing Pre Calibration Pipeline Files and Figures in Folder...\n');

% Chose Path to Store Folders
base_path = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
                     'Code', 'Matlab', 'Data', 'AutoPipeline', 'v1');           % !! CHANGE IF NEW VERSION !!

% Construct folder names
samples_data_name = sprintf('%dsamples_data', opts.total_samples);
samples_main_name = sprintf('%dsamples_%dPct_Calib', opts.total_samples, opts.calib_ratio*100);

data_folder = fullfile(base_path, samples_data_name);
main_folder = fullfile(base_path, samples_main_name);
figures_folder = fullfile(main_folder, 'figures');

% Create folders if they don't exist
if ~exist(data_folder, 'dir'), mkdir(data_folder); end
if ~exist(main_folder, 'dir'), mkdir(main_folder); end
if ~exist(figures_folder, 'dir'), mkdir(figures_folder); end

% Get all open figure handles
figs = findall(0, 'Type', 'figure');

% Loop and save each figure with title-based filename
for i = 1:numel(figs)
    fig = figs(i);
    figure(fig); % Bring to foreground just in case

    % Try to get title from the active axis
    ax = get(fig, 'CurrentAxes');
    title_text = '';
    if ~isempty(ax)
        title_obj = get(ax, 'Title');
        if isprop(title_obj, 'String') && ~isempty(title_obj.String)
            title_text = title_obj.String;
            if iscell(title_text), title_text = strjoin(title_text); end
        end
    end

    % Clean title for filename (preserve readability)
    if ~isempty(title_text)
        clean_title = regexprep(title_text, '[^\w\s-]', '');          % remove symbols like : ( )
        clean_title = strtrim(regexprep(clean_title, '\s+', '_'));    % replace spaces with single underscore
        filename_base = sprintf('fig_%d_%s', i, clean_title);
    else
        filename_base = sprintf('figure_%d', i);
    end

    % Save as MATLAB FIG file
    savefig(fig, fullfile(figures_folder, [filename_base '.fig']));

    % Save as PNG
    saveas(fig, fullfile(figures_folder, [filename_base '.png']));
end

% Move all .mat files (except excel + metadata)
mat_files = dir('*.mat');

% Loop over and move dataset-specific .mat files into subfolders
for k = 1:length(mat_files)
    fname = mat_files(k).name;

    % Skip metadata file
    if contains(fname, sprintf('pipeline_metadata_%dsamples.mat', opts.total_samples))
        continue;
    end

    moved = false;
    for d = 1:numel(datasets)
        ds = datasets{d};
        if contains(fname, ds)
            ds_folder = fullfile(data_folder, ds);
            if ~exist(ds_folder, 'dir'), mkdir(ds_folder); end
            movefile(fname, fullfile(ds_folder, fname));
            moved = true;
            break;
        end
    end

    % If not dataset-specific, move to base data_folder
    if ~moved
        movefile(fname, fullfile(data_folder, fname));
    end
end

% Move the data folder into the main folder
movefile(data_folder, fullfile(main_folder, samples_data_name));

% Move the two Excel output files into the main folder
movefile(sprintf('PreCalib_%dsamples_stats.xlsx', opts.total_samples), ...
         fullfile(main_folder, sprintf('PreCalib_%dsamples_Stats.xlsx', opts.total_samples)));
movefile(opts.results_name, fullfile(main_folder, opts.results_name));

% Move metadata .txt and .mat files into the main folder as well
movefile(sprintf('pipeline_metadata_%dsamples.txt', opts.total_samples), ...
         fullfile(main_folder, sprintf('pipeline_metadata_%dsamples.txt', opts.total_samples)));
movefile(sprintf('pipeline_metadata_%dsamples.mat', opts.total_samples), ...
         fullfile(main_folder, sprintf('pipeline_metadata_%dsamples.mat', opts.total_samples)));

% Add the new Folder Path for usage in following Automated Calibration  Phase           % !! CHANGE IF NEW VERSION  !!
addpath(genpath(fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', 'Code', ...
    'Matlab', 'Data', 'AutoPipeline', 'v1', sprintf('%dsamples_%dPct_Calib', opts.total_samples, opts.calib_ratio*100))));

close all;

% -------------------------------------------------------------------------
% OFFLINE PIPLINE DONE
% -------------------------------------------------------------------------
fprintf('\n============================================\n')
disp('[DONE] Automated Offline Pipeline Completed.');



%% Automatic Transfer Learning with/without Domain Adaptation using Standard, Normalized and respective Hyperparameter Tuned Models

% -------------------------------------------------------------------------
% MASTER LOOP FOR ALL CALIBRATION CASES
% -------------------------------------------------------------------------
fprintf('\n\n [STAGE 8] Running Cross-Data Model Calibration and Evaluation... \n');

% -------------------------------------------------------------------------
% GENERAL PARAMETERS AND INITIALIZATIONS
% -------------------------------------------------------------------------
params                           = struct();
params.verbose                   = opts.verbose;
params.total_samples             = opts.total_samples;
params.calib_samples             = opts.calib_samples;
params.calib_ratio               = opts.calib_ratio;
params.epochlength               = opts.epochlength;
params.num_features              = opts.num_features;
params.num_csp_filters           = opts.num_csp_filters;
params.handcrafted_feature_names = opts.handcrafted_feature_names;
params.csp_feature_names         = opts.csp_feature_names;
params.combined_feature_names    = opts.combined_feature_names;

% Define Calib Types to iterate through
calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};

% Source vs Calibration Samples - File Naming Tag
calib_sample_tag = sprintf('AfterCalib_%dTotal_%dCalib_Results', params.total_samples, params.calib_samples);

% Results File Name
calib_matfile   = [calib_sample_tag  '.mat'];
calib_excelfile = [calib_sample_tag  '.xlsx'];

% Config Specific Initializations:
num_configs = numel(feature_configs);
after_calib_within_results_all = cell(1, num_configs);

for cfg = 1:num_configs

    % For Each Config:
    % Initializing "After Calibration Results" Table for New Model
    % Performance (After) Calibration -> Within and Cross Accuracies For each Config
    after_calib_within_results_all{cfg} = init_results_table(datasets, calib_types);

    % Parallel matrix to hold cross-acc
    after_calib_cross_results_all{cfg} = NaN(height(after_calib_within_results_all{cfg}),...
        width(after_calib_within_results_all{cfg})); 

    % Create Config specific Sheets (Hardcoded to pre create these sheets) 
    config_sheetname = sprintf('CalibResults_%s', string({feature_configs{cfg}.label}));

    % Write "CalibResults" as the first sheet right away
    writetable(after_calib_within_results_all{cfg}, calib_excelfile, 'WriteRowNames', true, 'Sheet', config_sheetname);
end

% -------------------------------------------------------------------------
% CALIBRATION LOOP
% -------------------------------------------------------------------------
for src = 1:length(datasets)
    for tgt = 1:length(datasets)
        if src == tgt, continue; end % Skip self-calibration

        for cfg = 1:numel(feature_configs)
            
            for hyper = [false true]
                for calib_type = calib_types

                    % -------------------------------------------------------------------------
                    % Setup Calibration Params and Calibration Flags
                    % -------------------------------------------------------------------------
                    % Source Dataset
                    params.dataset         = datasets{src};
                    params.proc            = proc_types.(params.dataset);
                    params.version         = versions.(params.dataset);

                    % Cross Dataset
                    params.calibrationset  = datasets{tgt};
                    params.cross_proc      = proc_types.(params.calibrationset);
                    params.cross_version   = versions.(params.calibrationset);

                    % Feature Config Parameters
                    params.hyper           = hyper;
                    params.use_features    = feature_configs{cfg}.use_features;
                    params.use_csp         = feature_configs{cfg}.use_csp;

                    % Set Calibration Flags / Logic for Calibration Types
                    ct = calib_type{1};
                    params.only_domain_adaptation = strcmp(ct, 'adapted');
                    params.do_transfer_learning   = strcmp(ct, 'finetuned') || strcmp(ct, 'finetuned_adapted');
                    params.do_domain_adaptation   = strcmp(ct, 'finetuned_adapted');

                    fprintf('\n>> %s → %s | %s | Hyper: %s\n', ...
                        params.dataset, params.calibrationset, ct, string(params.hyper));


% -------------------------------------------------------------------------
% RUN CALIBRATION
% -------------------------------------------------------------------------
                    try
                        [acc1, acc2, calib_info, params, per_class_table_source, per_class_table_cross] = run_calibration(params);
                    catch ME
                        warning('[FAILED] %s → %s (%s) | Hyper: %s\n%s', ...
                            params.dataset, params.calibrationset, ct, string(hyper), ME.message);
                        acc1 = NaN; acc2 = NaN;
                        calib_info = struct('samples', params.calib_samples, 'source_total', NaN, 'ratio', NaN);
                    end


% -------------------------------------------------------------------------
% Update and Save Current Calibration Results
% -------------------------------------------------------------------------
                    % Update Results Table
                    after_calib_within_results_all{cfg} = write_to_results_table(after_calib_within_results_all{cfg}, params, acc1, acc2, calib_info);

                    % Save Accuracies into matrix form
                    row_idx = find(strcmp(after_calib_within_results_all{cfg}.Properties.RowNames, ...
                        sprintf('%s (Hyper: %s)', params.dataset, upper(string(params.hyper)))));
                    col_idx = find(strcmp(after_calib_within_results_all{cfg}.Properties.VariableNames, ...
                        matlab.lang.makeValidName(sprintf('%s (%s)', params.calibrationset, params.calibration))));
                    after_calib_cross_results_all{cfg}(row_idx, col_idx) = acc2;   % Store cross accuracy

                    % Generate Model Label (STANDARD, HYPER, NORM, HYPER NORM)
                    if ~params.hyper && strcmp(params.modeltype, 'model')
                        calib_model_label = 'STANDARD';
                    elseif params.hyper && strcmp(params.modeltype, 'model')
                        calib_model_label = 'HYPER';
                    elseif ~params.hyper && strcmp(params.modeltype, 'norm_model')
                        calib_model_label = 'NORM';
                    elseif params.hyper && strcmp(params.modeltype, 'norm_model')
                        calib_model_label = 'HYPER NORM';
                    else
                        calib_model_label = 'UNKNOWN';
                    end

                    calib_source_dataset_tag = sprintf('%s %s', params.dataset, feature_configs{cfg}.label);
                    calib_cross_dataset_tag = sprintf('%s %s', params.calibrationset, feature_configs{cfg}.label);

                    % Save Per Class Behavior
                    % Class-specific (LOW/ HIGH) Evaluation on the Source dataset (e.g., STEW after fine-tuning)
                    % = "How is the model performing on the source data after calibration?"
                    if ~isempty(per_class_table_source)
                        write_class_metrics_calibration_excel(per_class_table_source, calib_source_dataset_tag, ...
                            'CalibrationSource (WITHIN)', calib_model_label, params.calibration, calib_excelfile);
                    end

                    % Class-specific (LOW/ HIGH) Evaluation on the Cross dataset (e.g., MATB after fine-tuning)
                    % = "How is the model performing on the cross data after calibration?"
                    if ~isempty(per_class_table_cross)
                        write_class_metrics_calibration_excel(per_class_table_cross, calib_source_dataset_tag, ...
                            calib_cross_dataset_tag, calib_model_label, params.calibration, calib_excelfile);
                    end
                end
            end

            %Write (config-specific) Results
            config_tag = feature_configs{cfg}.label;
            config_sheetname = ['CalibResults_' config_tag];
            writetable(after_calib_within_results_all{cfg}, calib_excelfile, ...
                'WriteRowNames', true, 'Sheet', config_sheetname);
        end
    end

    % After finishing all target/calibration combinations for specific
    % source - sort the class sheets after Calibration Types
    if startsWith(params.dataset, 'MATB_easy_med')
        dataset_name = 'EasyMedDiff';
    elseif startsWith(params.dataset, 'MATB_easy_diff')
        dataset_name = 'EasyDiff';
    else 
        dataset_name = params.dataset;
    end

    class_sheet_name = sprintf('Classes_%s', dataset_name);
    sort_class_metrics_sheet(calib_excelfile, class_sheet_name);
end

% Save Total Calibration Results to .mat file
save(calib_matfile, 'after_calib_within_results_all', 'after_calib_cross_results_all');
fprintf('[SAVED] %s and %s\n', calib_matfile, calib_excelfile);


% -------------------------------------------------------------------------
% SUMMARY AND STATISTICAL EVALUATION OF CALIBRATION RESULTS
% -------------------------------------------------------------------------
fprintf('\n\n [STAGE 9] Performing Statistical Evaluation of Calibration Results... \n');

base_feats = sprintf('%d', params.num_features);
combined_feats = sprintf('%dwCsp', params.num_features);
config_tags = {base_feats, 'csp', combined_feats}; 

% Extract the Best Performing Calibration results
extract_top_calibration_results(calib_excelfile, config_tags, 75);              % Only Take Model Performance above Threshold   

% Get Delta Accuracy PRE vs POST Calibration
% Extract Only PRE Calibration Cross Data as a Table
T_pre_cross_accuracies = extract_precalib_table(opts.results_name);

% Now Extract Only POST Calibration Cross Data as a Table 
T_post_cross_accuracies = extract_postcalib_table(calib_excelfile, config_tags);

% Load both Pre and Post Cross Accuracies
load('PreCalib_Cross_Results.mat');   % loads T_pre_cross_accuracies
load('PostCalib_Cross_Results.mat');  % loads T_post_cross_accuracies

% Compute Deltas
T_delta = compute_calibration_deltas(T_pre_cross_accuracies, T_post_cross_accuracies);

% Statistical Evaluation
compute_calibration_stats(calib_excelfile, params, config_tags);

% Move Delta Sheet into Stats File
move_delta_into_stats(params);



% -------------------------------------------------------------------------
% AUTOMATIC STORAGE
% -------------------------------------------------------------------------
store_calibration_pipeline_files(params, datasets);                             % !! CHANGE VERSION FOLDER INSIDE IF NEEDED !!


% -------------------------------------------------------------------------
% DONE
% -------------------------------------------------------------------------
disp('[DONE] All Calibration Combinations Evaluated');
fprintf('\n============================================\n')
disp('[DONE] Calibration Phase Completed.');
fprintf('\n============================================\n')
fprintf('============================================\n')

disp('[✓] Total Pipeline Complete.');
