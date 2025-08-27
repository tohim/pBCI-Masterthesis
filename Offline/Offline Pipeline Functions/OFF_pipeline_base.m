%% LEGACY MANUAL OFFLINE PIPELINE (for reference)

% Kept in the Offline Code Folder bc the data preprocessing can be done here manually for each dataset
% Then later in the main and up to date "OFF_pipeline" the already preprocessed data is loaded, to avoid heavy
% preprocessing time when executing the main automation script 

%% Manual Offline Pipeline
% 
% % Pipeline Options
opts.total_samples = 1000;                  % Total Number of Samples to use from Dataset (must be even for class balancing)
opts.dataset = 'STEW';                      % Select Current Source Dataset
opts.version = '';                          % v1 (=> '') = 4sec & proc3wRef or proc3noRef;   || Version 2 -> write "_v2"
opts.epochlength = '4sec';                  % Name for Epoch Length 
opts.proc = 'proc5';                   % with or without Refercing (if done already)
opts.use_features = true;                   % Use feature base
opts.num_features = 25;                     % Amount of Features
opts.use_csp = true;                        % Activate CSP Features
opts.num_csp_filters = 6;                   % Amount of CSP Filters
% opts.cross_dataset = 'HEATCHAIR';           % Select current Cross Dataset !Only compare Datasets with the same types of features!
% opts.cross_version = '';                    
% 
% % Variables
opts.fs = 128;                                                  % Sampling Frequency
opts.epoch_length = 4*opts.fs;                                  % Length of 1 Epoch in Seconds
opts.overlap = 0.5;                                             % Amount of Overlap between Epochs
opts.step_size = round((1-opts.overlap) * opts.epoch_length);   % Amount of Single Step Size in Samples
%
% % -------------------------------------------------------------------------
% % Load EEG Dataset (Segmented into Epochs with corresponding Labels)
% % -------------------------------------------------------------------------
clear; clc;
fprintf('\n\n [] Loading Segmented EEG and Variables: \n');
eeg_data_loader = load('4sec_raw_HEATCHAIR_epochs.mat');
eeg_data = eeg_data_loader.eeg_data;

% % -------------------------------------------------------------------------
% % Preprocess Epochs (Referencing, Filtering, ASR)
% % -------------------------------------------------------------------------
fprintf('\n\n [] (Loading) Preprocessing the EEG and Labels:  \n');
fs = 128;  
eeg_data_processed = OFF_preprocess_epochs(eeg_data,fs,false);               % true = Apply Common Average Referencing | false = No Common Average Referencing
%[eeg_data_processed, labels] = load_processed_data(opts); 



%%
% -------------------------------------------------------------------------
% Check Difference BEFORE vs AFTER Preprocessing (and if Common Average Referencing is applied):
% -------------------------------------------------------------------------

% (RE) Average Reference the dataset to check if Avg Referencing is needed:
for ep = 1:size(eeg_data_processed, 3)
    eeg_data_avgref(:,:,ep) = eeg_data_processed(:,:,ep) - mean(eeg_data_processed(:,:,ep), 1);
end

mean_across_channels = mean(eeg_data_avgref, 1);                       % [1 × Time × Epochs]
mean_across_channels_epochs = squeeze(mean(mean_across_channels, 3));  % [1 × Time]

% Load one epoch of STEW data (without re-referencing it again)
raw_ref_epoch     = eeg_data(:,:,5);                                % Taking a STEW epoch before i apply my preprocessing 
oneTime_ref_epoch = eeg_data_processed(:,:,5);                      % It says "processed", but here is no avg referencing applied
twoTime_ref_epoch = eeg_data_avgref(:,:,5);                         % Here is now avg referencing applied (so a "2nd Time" because study states it already did avg referencing)

% Compute mean across channels for each sample: 1 x N_samples
raw_channel_mean     = mean(raw_ref_epoch,1);
oneTime_channel_mean = mean(oneTime_ref_epoch, 1);  
twoTime_channel_mean = mean(twoTime_ref_epoch, 1);

% Compute overall deviation from zero
raw_mean_offset     = mean(raw_channel_mean);
raw_std_offset      = std(raw_channel_mean);

oneTime_mean_offset = mean(oneTime_channel_mean);
oneTime_std_offset  = std(oneTime_channel_mean);

twoTime_mean_offset = mean(twoTime_channel_mean);
twoTime_std_offset  = std(twoTime_channel_mean);

fprintf('One Time Ref ("RAW FROM STUDY") Mean offset across time: %.5f\n', raw_mean_offset);
fprintf('One Time Ref ("RAW FROM STUDY") Std deviation of mean across channels: %.5f\n', raw_std_offset);

fprintf('One Time Ref ("MY PROCESSING") Mean offset across time: %.5f\n', oneTime_mean_offset);
fprintf('One Time Ref ("MY PROCESSING") Std deviation of mean across channels: %.5f\n', oneTime_std_offset);

fprintf('Two Time Ref ("RE REFERENCED") Mean offset across time: %.5f\n', twoTime_mean_offset);
fprintf('Two Time Ref ("RE REFERENCED") Std deviation of mean across channels: %.5f\n', twoTime_std_offset);

% Just compare one epoch
original           = eeg_data(:,:,5);
original_processed = eeg_data_processed(:,:,5);
avgref             = eeg_data_avgref(:,:,5);

figure;
subplot(3,1,1);
plot(original(1,:)); title('Original "RAW" - Channel 1');

subplot(3,1,2);
plot(original_processed(1,:)); title('Original Processed - Channel 1');

subplot(3,1,3);
plot(avgref(1,:)); title('RE Avg Referenced - Channel 1');
%%
% -------------------------------------------------------------------------
% Randomly Split Data into (70% Training, 15% Validation, 15% Test)
% -------------------------------------------------------------------------
fprintf('\n\n [] Splitting EEG and Labels into 70/15/15 Train, Val, Test: \n');
% Remember to change Version for specific dataset! (v1,v2,v3,etc.)
[train_epochs, val_epochs, test_epochs, train_labels, val_labels, test_labels] = split_data(eeg_data_processed, labels, opts);

% -------------------------------------------------------------------------
% Feature Extraction
% -------------------------------------------------------------------------
fprintf('\n\n [] Extracting Features: \n');
% Remember to change Feature Naming/ Amount (24,27,etc.)
[train_features, val_features, test_features] = extract_all_features(train_labels, val_labels, test_labels, train_epochs, val_epochs, test_epochs, opts);
fprintf('Total Feature Extraction complete! Features saved.\n');

% -------------------------------------------------------------------------
% Classification
% -------------------------------------------------------------------------
fprintf('\n\n [] Classification (Loading Train/Val/Test Features): \n');
train_and_eval_models(train_features, val_features, test_features, train_labels, val_labels, test_labels, opts);
fprintf('Training and Evaluation complete! Standard (+Hyper) and Normalized (+Hyper) saved.\n');

% -------------------------------------------------------------------------
% Cross Dataset Testing - PRE CALIBRATION
% -------------------------------------------------------------------------
fprintf('\n\n [] Starting Cross-Dataset Testing:\n')
legacy_cross_dataset_eval(opts);



%% Manual Transfer Learning with/without Domain Adaptation using Standard, Normalized and respective Hyperparameter Tuned Models

% % -------------------------------------------------------------------------
% % Settings for Calibration 
% % -------------------------------------------------------------------------
% First Run the opts_base.*, opts, version, proc_types and feature_configs initializations for the Automated Offline Pipeline.

% Data & Model Selection
% params.num_features              = 24;                                    % Amount of Features                    = '23', '24', '27'
% params.epochlength               = 4sec;                                  % Epoch Length                          = '2sec', '4sec', '6sec'
% params.dataset                   = 'STEW';                                % Source/ Main/ Training Dataset        = 'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'
% params.proc                      = proc_types.(params.dataset);
% params.version                   = versions.(params.dataset);
% params.calibrationset            = 'MATB_easy_diff';                      % Cross-Data Calibration Dataset = 'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'
% params.cross_proc                = proc_types.(params.calibrationset);
% params.cross_version             = versions.(params.calibrationset);
% params.num_csp_filters           = opts.num_csp_filters;
% params.handcrafted_feature_names = opts.handcrafted_feature_names;
% params.csp_feature_names         = opts.csp_feature_names;
% params.combined_feature_names    = opts.combined_feature_names;

% % Calibration Options
% params.total_samples             = opts.total_samples;    % Pass Amount of Total Samples
% params.calib_samples             = 360;                   % Specify the amount of Cross-Data Samples for Calibration
% params.hyper                     = true;                  % Using Hyperparameter Tuned Models for Calibration
% params.only_domain_adaptation    = false;                 % Normalize Incoming Data using Source Statistics
% params.do_domain_adaptation      = false;                 % Normalize Incoming Data using the Source Statistics and pass it on for Transfer Learning
% params.do_transfer_learning      = true;                  % Add the Incoming Data to the Source Data and Retrain Classifier

% -------------------------------------------------------------------------
% Run Single Calibration 
% -------------------------------------------------------------------------
% run_calibration(params);



