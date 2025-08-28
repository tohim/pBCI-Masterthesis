%% Processing Pipeline for Real-Time Experiment

% -------------------------------------------------------------------------
% "Real-Time pBCI Mental Workload Detection with Adaptive System Change"
% -------------------------------------------------------------------------


%% Structure hui?

% I.  Calibration Phase    (Collect User-specific Data & Train Final Subject-Adapted Model using Transfer Learning)
% II. Real-Time Experiment (Predict Mental Workload and adapt System and Robot Behavior accordingly)

% Workflow: TCP-Communication, Preprocessing, Feature Extraction and Classification Steps
% 1. Get EEG data via TCP Port from Simulink (including a 50% Buffer in Simulink = 4sec w 50% overlap = 1 epoch/2sec)
% 2. Preprocessing (Filtering (2-20 Hz Bandpass Filtering), Artifact Removal (MAD-Treshold, Channel Interpolation)
% 3. Feature Extraction (Handcrafted (Mainly Power and Brain Region based Features + Model-specific CSP Features)
% 4. Classification - 3 pre-trained Base Models feed predictions into FIFO Majority Vote Buffer for final Prediction
% 5. Classifier Output --> adaptive System Difficulty (easy/hard screen cue) & adaptive Robot Instructions sent to Python
% Script via TCP to execute different Robot Movement


%% Initialization
% write(t_py, uint8(command_stop));
clear; close all; clc; clear acquire_new_data;

%% ------------------------------------------------------------------------
% Simulink & Recording Setup
% -------------------------------------------------------------------------
% Start Simulink

% Find my current laptop ipv4 address
% open cmd: ipconfig

% Connect to Simulink via TCP to receive continous EEG stream
% TCP server Connect:

tcp_port = 50001;

fprintf('[TCP] Initializing EEG Listener on Port %d...\n', tcp_port);

tcp_server_simulink = tcpserver(tcp_port, ...
    "ConnectionChangedFcn", @(src, evt) fprintf('[TCP SIMULINK] Connection: %s\n', evt.Connected));


% Ping Matlab Laptop from Simulink Laptop to check if connection is working
% ping my ipv4 address


% Impedance Check

%% ------------------------------------------------------------------------
% Interface to Python Script -> Sending Robot Commands
% -------------------------------------------------------------------------
% Connect to Python Instance via TCP to send commands to execute robot
% scripts
% Create TCP/IP client to connect to Python server
t_py = tcpclient('127.0.0.1', 65432);

% Example:
% Send command ("LOW", "HIGH", or "STOP")
%command = 'LOW';  % or 'LOW' or 'STOP' / 'CONTINUE'
%write(t_py, uint8(command));

% write(t_py, uint8(command_stop));

% (Optional) Wait for Python acknowledgment (read what python sends back)
% data = read(t_py);
% disp(char(data));

% Close connection
% clear t

%% ------------------------------------------------------------------------
% Environment Settings
% -------------------------------------------------------------------------
state   = 'realtime';     % 'realtime' or 'testing' state
setting = 'adaptive';     % 'adaptive' or 'fixed' mode

% Define Protocol
eeg_protocol = 'tcp';   % or 'udp' or 'lsl'

% Handle missing tcp_server_simulink for testing
if strcmpi(state, "testing")
    tcp_server_simulink = [];  % Dummy placeholder
end

% -------------------------------------------------------------------------
% Parameters
% -------------------------------------------------------------------------
subject_number   = 11;                                    % Subject Number for Model Save and Logging - !! CHANGE !!

fs               = 128;                                  % Sampling Frequency 128 Hz
nbchan           = 14;                                   % Channel Number
channels         = {'AF3','F7','F3','FC5','T7','P7','PO7','PO8','P8','T8','FC6','F4','F8','AF4'}; % O1 and O2 not directly available, closest
%naut_ch_nmbrs   = {   3 , 5  , 6  , 10  , 14 , 23 , 28  , 31  , 27 , 18 , 13  , 8  , 9  , 4   }; % neighbors are PO7 and PO8
epoch_sec        = 4;                                    % Epoch Length in Seconds
epoch_length     = epoch_sec * fs;                       % Epoch Length in Samples
num_features     = 25;                                   % Set Number of Features based on Selected Version (25 Base) (+6 CSP Filters)
overlap          = 0.5;                                  % 50% overlap -> updated Classification Output every epoch_sec/2sec = every 2sec
MWL_buffer_valid = 60;                                   % Number of past Classification Outputs to consider (60/ 3 predictions at a time = 20 epochs)
MWL_buffer       = -1*ones(MWL_buffer_valid, 1);         % Initializes a buffer with -1 (invalid buffer values of size 60)
command_low      = 'LOW';                                % Robot Command to execute Low Workload Behavior
command_high     = 'HIGH';                               % Robot Command to execute High Workload Behavior
command_continue = 'CONTINUE';                           % Robot Command to allow to continue Robot Task Execution
command_stop     = 'STOP';                               % Robot Command to stop Robot Task Execution
[b,a]            = butter(2, [2 20]/(fs/2), 'bandpass'); % 2nd Order Butterworth Filter Parameters                                                      % LowCutoff:2 (Slow Drifts), HighCutoff:20 (rough EMG Artifacts))


% -------------------------------------------------------------------------
% Initialize Data Storage and Log
% -------------------------------------------------------------------------
experiment_log = struct();                      % Structure to store all experiment related information
calibration_log = struct();                     % Structure to store all calibration related information

% -------------------------------------------------------------------------
% Initialize Transfer Learning Data Storage and Parameters
% -------------------------------------------------------------------------
calib_epoch_idx = 1;                            % Initialize Calibration Epoch Log Index

calibration_data_STEW = [];                     % Storage for Calibration Data for STEW Model
calibration_data_HEAT = [];                     % Storage for Calibration Data for HEAT Model
calibration_data_MATB = [];                     % Storage for Calibration Data for MATB Model

calibration_labels_STEW = [];                   % Storage for Calibration Labels
calibration_labels_HEAT = [];                   % Storage for Calibration Labels
calibration_labels_MATB = [];                   % Storage for Calibration Labels

% Calibration Settings
blocks_per_class = 10;                          % Divide into 10 Blocks (60sec/block) 
% only 4 blocks are used as calibration data, rest for later post processing in a differenct project
samples_per_block = 30;                         % Take 30 Samples per Block (30*2sec)

% Amount of Calibration Samples = 240 --> ~30% of Training Data (~34% for STANDARD, ~28% for HYPER)
calibration_samples_needed = blocks_per_class * samples_per_block * 2;

% Define a weight factor for calibration samples
k = 1;      % "k-times the importance for new incoming user data"

% -------------------------------------------------------------------------
% Load Pre-Trained Classifiers
% -------------------------------------------------------------------------
% Load Pretrained STANDARD STEW SVM Model
mdl_name_stew = 'v1_Base_1000_25wCsp_4sec_proc5_STEW_model.mat';
loaded_model_stew = load(mdl_name_stew);
mdl_workload_STEW = loaded_model_stew.mdl;
W_csp_STEW = loaded_model_stew.W_csp;                     % Load Training Data Common Spatial Filters

% Load Pretrained STANDARD HYPERPARAMETER TUNED HEATCHAIR SVM Model
mdl_name_heat = 'v1_Base_hyper_1000_25wCsp_4sec_proc5_HEATCHAIR_model.mat';
loaded_model_heat = load(mdl_name_heat);
mdl_workload_HEAT = loaded_model_heat.mdl;
best_C_heat = loaded_model_heat.best_C;                   % Load Training Data Hyperparameter Best C
best_kernel_heat = loaded_model_heat.best_kernel;         % Load Training Data Hyperparameter Best Kernel
W_csp_HEAT = loaded_model_heat.W_csp;                     % Load Training Data Common Spatial Filters


% Load Pretrained STANDARD HYPERPARAMETER TUNED MATB SVM Model
mdl_name_matb = 'v1_Base_hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat';
loaded_model_matb = load(mdl_name_matb);
mdl_workload_MATB = loaded_model_matb.mdl;
best_C_matb = loaded_model_matb.best_C;                   % Load Training Data Hyperparameter Best C
best_kernel_matb = loaded_model_matb.best_kernel;         % Load Training Data Hyperparameter Best Kernel
W_csp_MATB = loaded_model_matb.W_csp;                     % Load Training Data Common Spatial Filters


% -------------------------------------------------------------------------
% Initialize Screen Cue Setup
% -------------------------------------------------------------------------
fig_cue = figure('Name', 'Cue Screen', 'NumberTitle', 'off');
cue_ctrl = CueController(fig_cue);  % object holding state


%% 1. CALIBRATION PHASE
fprintf('\n[START] Starting Calibration Phase.')

% -------------------------------------------------------------------------
% Start EEG Data Stream
% -------------------------------------------------------------------------
% Create randomized calibration block sequence
calibration_seed = rng(3);                                              % Set random seed for reproducibility
calibration_log_metadata.rng_seed = calibration_seed;                   % Save seed

block_labels = [zeros(1, blocks_per_class), ones(1, blocks_per_class)]; % Zeros = Low | Ones = High Labels for each Block
block_labels = block_labels(randperm(numel(block_labels)));             % Shuffle
calibration_log_metadata.block_labels = block_labels;

% -------------------------------------------------------------------------
% PARADIGM CALIBRATION PHASE:
% Each Block runs for 30 epochs (1 epoch every 2 sec) = 60 sec/Block
% Loop through a total of 20 Blocks (8 Blocks used for Calibration). 10 (4 for Calib) Blocks per Class Label (Low/High)
% 1 block = 30 samples * 8 Blocks = 240 Calibration samples.
% 20 blocks * 30 samples = 600 Total Samples collected/ subject.
% Total of 20 (8) Minutes of Calibration Phase.
% -------------------------------------------------------------------------

% 20 minutes for all subject
% then take the last 240 samples -> to stabilize percetual load/ getting accustomed to the task and paradigm

% Loop through each calibration block
for calib_block_idx = 1:length(block_labels)
    label = block_labels(calib_block_idx);

    while true % repeat paradigm until 'Press Enter to get to next Loop' or 'REDO'

        % Start task
        if label == 0                   % Low Workload
            command = command_low;      % If Label = 0 (=Low Workload); Send command_low to execute easy/slow behavior: "LOW"
        elseif label == 1               % High Workload
            command = command_high;     % If Label = 1 (=High Workload); Send command_high to execture difficult/fast behavior: "HIGH"
        end

        fprintf('\nStarting Calibration Block %d/%d | Label = %d (%s)\n', ...
            calib_block_idx, length(block_labels), label, command);

        figure(fig_cue);
        block_start_msg = sprintf('Waiting for \nConfirmation \nto Start.');
        cue_ctrl.showMessage(block_start_msg, 0);  % pause - Show adapting message

        %------------------- WAIT FOR USER TO START TRIAL ------------------------%
        input('Press ENTER to start the Calibration Block...\n');                 % Wait for Confirmation
        % ------------------------------------------------------------------------%

        % Send Robot Command to start Task
        write(t_py, uint8(command_continue));      % Allow to start/ continue command script
        write(t_py, uint8(command));               % write block command to python

        % -------------------------------------------------------------------------
        % Start Screen Paradigm for Calibration Phase (Color Cue)
        % -------------------------------------------------------------------------
        cue_ctrl.condition = command;
        cue_ctrl.resetCountdown();
        cue_ctrl.resetInternalTimer();
        cue_ctrl.stopFlag = false;
        cue_ctrl.startTimer();

        % -------------------------------------------------------------------------
        % Start Calibration Phase
        % -------------------------------------------------------------------------
        % Loop through 30 samples per block (1 samples = 4 sec epoch / 50% Overlap = 2seconds/samples = 60 sec)
        for s = 1:samples_per_block

            %pause(0.25); % for testing

            % -------------------------------------------------------------------------
            % Get Incoming Buffer Data
            % -------------------------------------------------------------------------
            % Get the Buffered Continous EEG Signal Segements from Simulink

            [epoch_data_acq, raw_pre_sample] = acquire_new_data(eeg_protocol, state, tcp_server_simulink);   % REALTIME: remove inputs: label & 'calib' !!!

            % Get EEG Stream Epoch Data in Format [14 x 512] with correctly
            % ordered channel numbers (matching Emotiv Epoch setup)
            epoch_data = order_channels(epoch_data_acq);

            % -------------------------------------------------------------------------
            % Preprocessing
            % -------------------------------------------------------------------------
            processed_epoch = RT_preprocess_epochs(epoch_data,fs,b,a);

            % -------------------------------------------------------------------------
            % Feature Extraction
            % -------------------------------------------------------------------------
            calib_handcrafted_epoch_features = RT_extract_features(processed_epoch, epoch_length, fs, num_features);
            calib_csp_features_stew          = extract_csp_features_single_epoch(processed_epoch, W_csp_STEW);
            calib_csp_features_heat          = extract_csp_features_single_epoch(processed_epoch, W_csp_HEAT);
            calib_csp_features_matb          = extract_csp_features_single_epoch(processed_epoch, W_csp_MATB);

            % Combine both Feature Vectors for each Model
            calib_epoch_features_STEW = [calib_handcrafted_epoch_features, calib_csp_features_stew];
            calib_epoch_features_HEAT = [calib_handcrafted_epoch_features, calib_csp_features_heat];
            calib_epoch_features_MATB = [calib_handcrafted_epoch_features, calib_csp_features_matb];

            % -------------------------------------------------------------------------
            % Calibration Data Storage (for Fine-Tuning the SVM Model)
            % -------------------------------------------------------------------------
            % Store Calibration Epochs and Labels
            calibration_data_STEW   = [calibration_data_STEW; calib_epoch_features_STEW];
            calibration_labels_STEW = [calibration_labels_STEW; label];

            calibration_data_HEAT   = [calibration_data_HEAT; calib_epoch_features_HEAT];
            calibration_labels_HEAT = [calibration_labels_HEAT; label];

            calibration_data_MATB   = [calibration_data_MATB; calib_epoch_features_MATB];
            calibration_labels_MATB = [calibration_labels_MATB; label];

            % -------------------------------------------------------------------------
            % Logging calibration epoch
            % -------------------------------------------------------------------------
            calibration_log(calib_epoch_idx).block           = calib_block_idx;             % Save Block Count 
            calibration_log(calib_epoch_idx).sample_in_block = s;                           % Save EEG sample within block
            calibration_log(calib_epoch_idx).true_label      = label;                       % Save the true labels
            calibration_log(calib_epoch_idx).full_raw        = raw_pre_sample;              % Truly Raw EEG without resampling/ downsampling (250 Hz)
            calibration_log(calib_epoch_idx).raw             = epoch_data_acq;              % Save Raw Calibration Data from Nautilus (128 Hz)
            calibration_log(calib_epoch_idx).raw_ordered     = epoch_data;                  % Save Raw Calibration EEG in Emotiv Epoch Ch-Setup
            calibration_log(calib_epoch_idx).processed       = processed_epoch;             % Save Processed Epochs
            calibration_log(calib_epoch_idx).features_STEW   = calib_epoch_features_STEW;   % Save Stew features
            calibration_log(calib_epoch_idx).features_HEAT   = calib_epoch_features_HEAT;   % Save Heat features
            calibration_log(calib_epoch_idx).features_MATB   = calib_epoch_features_MATB;   % Save Matb features
            calibration_log(calib_epoch_idx).timestamp       = datetime('now');             % Save time and date of the measurement

            calib_epoch_idx = calib_epoch_idx + 1;

            fprintf('→ Block %2d | Sample %2d/%d Processed and stored.\n', calib_block_idx, s, samples_per_block);

        end

        % Robot Stop Task
        write(t_py, uint8(command_stop));

        % Stop Cue Screen
        cue_ctrl.stopFlag = true;
        stop(cue_ctrl.timerObj);

        % Starting Screen
        figure(fig_cue);
        block_done_msg = sprintf('Block %d / %d Done.\n', calib_block_idx, length(block_labels));
        cue_ctrl.showMessage(block_done_msg, 0);  % pause - Show adapting message

        % Enter "REDO" to repeat Block or "STOP" to end the Calibration Loop or press "Enter" to continue
        end_experiment_input = input('Press ENTER to continue, type "REDO" to repeat Block, or "STOP" to end: ', 's'); % Waits for User Input
        if strcmpi(end_experiment_input, 'STOP')
            write(t_py, uint8(command_stop));
            fprintf('[STOP] Calibration manually stopped by user.\n');
            return;                                      % Exerpiment Permanently Interrupted -> Exit Loop
        elseif strcmpi(end_experiment_input, 'REDO')
            fprintf('[REDO] Repeating current calibration Block %d...\n', calib_block_idx);

            % Remove last block's data (30 samples) from all calibration arrays
            calibration_data_STEW(end-samples_per_block+1:end, :) = [];
            calibration_labels_STEW(end-samples_per_block+1:end) = [];

            calibration_data_HEAT(end-samples_per_block+1:end, :) = [];
            calibration_labels_HEAT(end-samples_per_block+1:end) = [];

            calibration_data_MATB(end-samples_per_block+1:end, :) = [];
            calibration_labels_MATB(end-samples_per_block+1:end) = [];

            % Remove from calibration log
            calib_epoch_idx = calib_epoch_idx - samples_per_block;
            calibration_log(calib_epoch_idx+1:end) = [];
            continue;

        else
            break;
        end
    end
end

% Save Calibration Log and Metadata
save_name = sprintf('Subject%d_CalibrationLog.mat', subject_number);
save(save_name, 'calibration_log', 'calibration_log_metadata');

fprintf('\n[INFO] Calibration Data Collection Complete! Performing Transfer Learning...\n');


% -------------------------------------------------------------
% Use only the last 240 calibration samples for this project
% -------------------------------------------------------------
% The rest of the calibration data will be used in later projects.
% This equivalent to 8 min of calibration data is kept for simulated
% real-world implementation
% Taking only the last 240 samples of more "mature" data where subjects
% are already better used to the task

% Set how many samples to keep
calibration_samples_keep = 240;

% Keep only the last 240 samples for each model
calibration_data_STEW = calibration_data_STEW(end - calibration_samples_keep + 1:end, :);
calibration_labels_STEW = calibration_labels_STEW(end - calibration_samples_keep + 1:end);

calibration_data_HEAT = calibration_data_HEAT(end - calibration_samples_keep + 1:end, :);
calibration_labels_HEAT = calibration_labels_HEAT(end - calibration_samples_keep + 1:end);

calibration_data_MATB = calibration_data_MATB(end - calibration_samples_keep + 1:end, :);
calibration_labels_MATB = calibration_labels_MATB(end - calibration_samples_keep + 1:end);

% Safety Check in case of label unbalance
labels = calibration_labels_STEW;  % Or any other model – labels are the same

num_low = sum(labels == 0);
num_high = sum(labels == 1);

fprintf('[INFO] Calibration set contains:\n');
fprintf('  LOW MWL  (label = 0): %d samples\n', num_low);
fprintf('  HIGH MWL (label = 1): %d samples\n', num_high);


% -------------------------------------------------------------------------
% Transfer Learning (Fine-tuning the Model with newly acquired Calibration Data)
% -------------------------------------------------------------------------

% Load Base Training and Calibration Data into "all" matrices
X_all_STEW = [mdl_workload_STEW.X; calibration_data_STEW];
Y_all_STEW = [mdl_workload_STEW.Y; calibration_labels_STEW];

X_all_HEAT = [mdl_workload_HEAT.X; calibration_data_HEAT];
Y_all_HEAT = [mdl_workload_HEAT.Y; calibration_labels_HEAT];

X_all_MATB = [mdl_workload_MATB.X; calibration_data_MATB];
Y_all_MATB = [mdl_workload_MATB.Y; calibration_labels_MATB];

% Build weight vectors weight = 1 for base samples | weight = k for calibration samples
% Weight Matrix Base of size [TrainingData x 1] and Calib of size [CalibData x 1]
w_base_STEW = ones(size(mdl_workload_STEW.Y));
w_calib_STEW = k * ones(size(calibration_labels_STEW));
weights_all_STEW = [w_base_STEW; w_calib_STEW];

w_base_HEAT = ones(size(mdl_workload_HEAT.Y));
w_calib_HEAT = k * ones(size(calibration_labels_HEAT));
weights_all_HEAT = [w_base_HEAT; w_calib_HEAT];

w_base_MATB = ones(size(mdl_workload_MATB.Y));
w_calib_MATB = k * ones(size(calibration_labels_MATB));
weights_all_MATB = [w_base_MATB; w_calib_MATB];

% Retrain SVM with (optionally) weighted samples:

% Transfer Learning on STANDARD STEW dataset
Calib_mdl_workload_STEW = fitcsvm(X_all_STEW, Y_all_STEW, ...
    'KernelFunction',    'linear', ...
    'BoxConstraint',     1,  ...
    'Weights',           weights_all_STEW ...
    );
% Save Calibrated Model for current Subject
calib_mdl_name_stew = sprintf('Subject%d_CalibratedModel_STEW.mat', subject_number);
save(calib_mdl_name_stew, 'Calib_mdl_workload_STEW', 'W_csp_STEW');


% Transfer Learning on HYPER HEAT dataset
Calib_mdl_workload_HEAT = fitcsvm(X_all_HEAT, Y_all_HEAT, ...
    'KernelFunction',    best_kernel_heat, ...
    'BoxConstraint',     best_C_heat,  ...
    'Weights',           weights_all_HEAT ...
    );
% Save Calibrated Model for current Subject
calib_mdl_name_heat = sprintf('Subject%d_CalibratedModel_HEAT.mat', subject_number);
save(calib_mdl_name_heat, 'Calib_mdl_workload_HEAT', 'W_csp_HEAT');


% Transfer Learning on HYPER MATB Easy Diff dataset
Calib_mdl_workload_MATB = fitcsvm(X_all_MATB, Y_all_MATB, ...
    'KernelFunction',    best_kernel_matb, ...
    'BoxConstraint',     best_C_matb, ...
    'Weights',           weights_all_MATB ...
    );
% Save Calibrated Model for current Subject
calib_mdl_name_matb = sprintf('Subject%d_CalibratedModel_MATB.mat', subject_number);
save(calib_mdl_name_matb, 'Calib_mdl_workload_MATB', 'W_csp_MATB');

fprintf('\n[INFO] Fine-Tuning complete. Calibrated Model Saved.\n');


% -------------------------------------------------------------------------
% Save Calibration Summary Log (.txt)
% -------------------------------------------------------------------------
calib_summary_filename = sprintf('Subject%d_CalibrationSummary.txt', subject_number);
fid = fopen(calib_summary_filename, 'w');

fprintf(fid, '--- Calibration Phase Summary ---\n');
fprintf(fid, 'Subject ID         : %d\n', subject_number);
fprintf(fid, 'Date & Time        : %s\n', datetime("now"));
fprintf(fid, 'Model 1            : %s\n', mdl_name_stew);
fprintf(fid, 'Model 2            : %s\n', mdl_name_heat);
fprintf(fid, 'Model 3            : %s\n', mdl_name_matb);
fprintf(fid, '# Channels         : %d\n', nbchan);
fprintf(fid, 'Selected Channels  : %s\n', strjoin(channels, ', '));
fprintf(fid, 'Epoch Length       : %.1f sec\n', epoch_sec);
fprintf(fid, 'Sampling Rate      : %d Hz\n', fs);
fprintf(fid, 'Total Blocks       : %d\n', length(block_labels));
fprintf(fid, 'Samples per Block  : %d\n', samples_per_block);
calib_label_string = sprintf('%d', block_labels);
fprintf(fid, 'Block Label Order  : %s\n', calib_label_string);
fprintf(fid, 'Total Epochs       : %d\n', length(calibration_log));

% Count class labels
labels = [calibration_log.true_label];
fprintf(fid, 'Low MWL Epochs     : %d\n', sum(labels == 0));
fprintf(fid, 'High MWL Epochs    : %d\n', sum(labels == 1));

% Log final training set stats
fprintf(fid, '\n--- Model Re-training Stats ---');
fprintf(fid, '\n------------ STEW -------------\n');
fprintf(fid, 'New Training Set Size: %s Source Samples + %d Calibration Samples\n', '700', size(calibration_data_STEW,1));
new_train_size = 700 + size(calibration_data_STEW,1);
calib_pct_stew = round(size(calibration_data_STEW,1) / 700 * 100);
fprintf(fid, 'New Training Set Size: %d  -> %.2f%% Calibration Samples', new_train_size, calib_pct_stew);
fprintf(fid, '\n');
fprintf(fid, '\n------------ HEAT -------------\n');
fprintf(fid, 'New Training Set Size: %s Source Samples + %d Calibration Samples\n', '850', size(calibration_data_HEAT,1));
new_train_size = 850 + size(calibration_data_HEAT,1);
calib_pct_heat = round(size(calibration_data_HEAT,1) / 850 * 100);
fprintf(fid, 'New Training Set Size: %d  -> %.2f%% Calibration Samples', new_train_size, calib_pct_heat);
fprintf(fid, '\n');
fprintf(fid, '\n------------ MATB -------------\n');
fprintf(fid, 'New Training Set Size: %s Source Samples + %d Calibration Samples\n', '850', size(calibration_data_MATB,1));
new_train_size = 850 + size(calibration_data_MATB,1);
calib_pct_matb = round(size(calibration_data_MATB,1) / 850 * 100);
fprintf(fid, 'New Training Set Size: %d -> %.2f%% Calibration Samples', new_train_size, calib_pct_matb);

fclose(fid);

fprintf('[DONE] Finished Calibration Phase.\n')
fprintf('__________________________________\n');


%% 2. REAL-TIME EXPERIMENT

% -------------------------------------------------------------------------
% PARADIGM REAL TIME EXPERIMENT:
% Running a Total of 10 Blocks with known Ground Truth. (5 Low and 5 High Blocks shuffled)
% 1 Block = 30 samples (1 sample = 2sec (50%Overlap)) = 60 sec/Block
% Filling Majority Vote Buffer with 3 predictions (1 per Model) per Epoch.
% When Majority Vote Buffer has equal or more than 60 entries: Turns Valid. (= After 20 Epochs / 40sec)
% After 20 Epochs (40 Seconds) make Final Prediction based on Majority Vote.

% Send the respective Command to Robot -> Change Behavior (for last 20 seconds).
% Then Pause. Robot Stops. New Block Starts after User Input.
% Total of 10 Minutes of Real Time Experiment Phase.
% -------------------------------------------------------------------------

fprintf('\n\n[START] Starting Real-Time Experiment.\n');

% -------------------------------------------------------------------------
% Live Plot Setup
% -------------------------------------------------------------------------
clear epoch_idx
epoch_idx = 1;     % Counting up for each finished epoch (after MWL output for individual epoch)

live_mwl_fig = figure('Name', 'Live MWL Prediction', 'NumberTitle', 'off');
hold on;
grid on;
xlabel('Epoch #');
ylabel('MWL Prediction');
ylim([-0.2 1.2]);
xlim([0 100]);

% Plot Lines
h_pred_STEW = plot(nan, nan, '--o', 'Color', [0 0.447 0.741], 'LineWidth', 0.25, 'DisplayName', 'STEW Prediction');       % Individual predictions from STEW
h_pred_HEAT = plot(nan, nan, '--o', 'Color', [0.850 0.325 0.098], 'LineWidth', 0.25, 'DisplayName', 'HEAT Prediction');   % Individual predictions from HEAT
h_pred_MATB = plot(nan, nan, '--o', 'Color', [0.494 0.184 0.556], 'LineWidth', 0.25, 'DisplayName', 'MATB Prediction');   % Individual predictions from MATB
h_maj_shadow = plot(nan, nan, '-x', 'Color', [0 0.6 0], 'LineWidth', 1, 'DisplayName', 'Ground Truth');    % Placeholder Ground Truth (shadow/fake)
h_maj = plot(nan, nan, '-x', 'Color', [0 0.6 0], 'LineWidth', 5, 'DisplayName', 'Majority Vote');          % Majority Vote

legend;

% -------------------------------------------------------------------------
% Local Parameter Setup
% -------------------------------------------------------------------------
% Prediction history
pred_history_STEW = [];
pred_history_HEAT = [];
pred_history_MATB = [];
majority_history  = [];
shadow_history    = [];

% Generate randomized workload sequence: 5 Low (0) and 5 High (1)
seed = rng(7);  % Seed RNG for randomness
num_blocks = 10;
sequence = [zeros(1,5), ones(1,5)];
sequence = sequence(randperm(num_blocks));  % Shuffle the sequence
experiment_log_metadata.rng_seed = seed;
experiment_log_metadata.sequence = sequence;

% Load Calibrated Models
calib_mdl_name_stew     = sprintf('Subject%d_CalibratedModel_STEW.mat', subject_number);
calib_model_loader_stew = load(calib_mdl_name_stew);
Calib_mdl_workload_STEW = calib_model_loader_stew.Calib_mdl_workload_STEW;
W_csp_STEW              = calib_model_loader_stew.W_csp_STEW;

calib_mdl_name_heat     = sprintf('Subject%d_CalibratedModel_HEAT.mat', subject_number);
calib_model_loader_heat = load(calib_mdl_name_heat);
Calib_mdl_workload_HEAT = calib_model_loader_heat.Calib_mdl_workload_HEAT;
W_csp_HEAT              = calib_model_loader_heat.W_csp_HEAT;

calib_mdl_name_matb     = sprintf('Subject%d_CalibratedModel_MATB.mat', subject_number);
calib_model_loader_matb = load(calib_mdl_name_matb);
Calib_mdl_workload_MATB = calib_model_loader_matb.Calib_mdl_workload_MATB;
W_csp_MATB              = calib_model_loader_matb.W_csp_MATB;


% Start Stopwatch
experiment_start_time = tic;

% Define Amount of Collected Epochs
epochs_per_block = 60 / epoch_sec * 2;    % Take 60 second period, divided by 4 sec epochs * 2 for 50% Overlap = 30 Epochs

% Starting Screen
figure(fig_cue);
block_start_msg = sprintf('Waiting for \nConfirmation \nto Start.');
cue_ctrl.showMessage(block_start_msg, 0);  % pause - Show adapting message


%------------------- WAIT FOR USER TO START EXPERIMENT -------------------%
input('Press ENTER to start the Block...\n');       % Wait for Confirmation
% ------------------------------------------------------------------------%

% -------------------------------------------------------------------------
% Start EEG Data Stream
% -------------------------------------------------------------------------

for rt_block_idx = 1:num_blocks

    while true % repeat paradigm until 'Press Enter to get to next Loop' or 'REDO'

        % Reset Adapted_epoch counter at start of block
        adapt_epoch = 0;

        % Run one Block: set label from block sequence and send initialization command
        ground_truth_label = sequence(rt_block_idx);    % 0 = LOW, 1 = HIGH
        if ground_truth_label == 0
            initial_command = 'LOW';
        else
            initial_command = 'HIGH';
        end

        fprintf('\n\nStarting Block %d | Label = %s\n', rt_block_idx, initial_command);

        % Mark Block Start in the plot
        figure(live_mwl_fig);
        xline(epoch_idx-1, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 3, 'DisplayName', '', 'HandleVisibility', 'off');

        % Add label annotation above the line
        figure(live_mwl_fig);
        block_text = sprintf('Block %d', rt_block_idx);
        text(epoch_idx, 1.1, [block_text, 'Label = ', initial_command], ...
            'FontSize', 12, 'HorizontalAlignment', 'left', 'Color', [0.3 0.3 0.3]);

        % Real robot command or simulation message
        if strcmpi(state, 'realtime')
            write(t_py, uint8(command_continue));
            write(t_py, uint8(initial_command));
            fprintf('[START] Robot receives command: %s\n', initial_command);
        else
            fprintf('[TEST] Simulated robot receives: %s\n', initial_command);
        end

        % Show task cue based on ground truth before starting block
        figure(fig_cue);
        cue_ctrl.condition = initial_command;
        cue_ctrl.resetCountdown();
        cue_ctrl.resetInternalTimer();
        cue_ctrl.stopFlag = false;
        cue_ctrl.startTimer();

        % Start block EEG collection
        block_start = tic;
        already_corrected = false;

        for ep = 1:epochs_per_block     % Collect for 60 seconds per block w 50% overlap  (60 seconds Low | High randomly iterating)
            % After 40 seconds make final Prediction and send Adaptation Command - Then 10 second Pause - then new block starts  

            % Calculate Epoch Start Time relative to Experiment Start
            epoch_start_time = toc(experiment_start_time); % Seconds since experiment started

            % -------------------------------------------------------------------------
            % Get Incoming Data Storage Buffer
            % -------------------------------------------------------------------------
            % Get EEG Stream Data in Format [14 x 512] from Simulink
            [epoch_data_acq, raw_pre_sample_rt] = acquire_new_data(eeg_protocol, state, tcp_server_simulink);   % REALTIME: remove label & 'rt' !!!

            % Get EEG Stream Epoch Data in Format [14 x 512] with correctly
            % ordered channel numbers (matching Emotiv Epoch setup)
            epoch_data = order_channels(epoch_data_acq);

            % -------------------------------------------------------------------------
            % Preprocessing
            % -------------------------------------------------------------------------
            processed_epoch = RT_preprocess_epochs(epoch_data,fs,b,a);

            % -------------------------------------------------------------------------
            % Feature Extraction
            % -------------------------------------------------------------------------
            base_epoch_features = RT_extract_features(processed_epoch, epoch_length, fs, num_features);
            csp_features_stew   = extract_csp_features_single_epoch(processed_epoch, W_csp_STEW);
            csp_features_heat   = extract_csp_features_single_epoch(processed_epoch, W_csp_HEAT);
            csp_features_matb   = extract_csp_features_single_epoch(processed_epoch, W_csp_MATB);

            % Combine both Feature Vectors for each Model
            epoch_features_STEW = [base_epoch_features, csp_features_stew];
            epoch_features_HEAT = [base_epoch_features, csp_features_heat];
            epoch_features_MATB = [base_epoch_features, csp_features_matb];

            % -------------------------------------------------------------------------
            % Classification using the updated mdl_workload model
            % -------------------------------------------------------------------------
            [MWL_stew, score_stew] = predict(Calib_mdl_workload_STEW, epoch_features_STEW);
            [MWL_heat, score_heat] = predict(Calib_mdl_workload_HEAT, epoch_features_HEAT);
            [MWL_matb, score_matb] = predict(Calib_mdl_workload_MATB, epoch_features_MATB);

            % -------------------------------------------------------------------------
            % Majority Buffer to make final MWL Prediction
            % -------------------------------------------------------------------------
            % Add a Buffer with 60 slots (MWL_buffer_valid) to collect the Classification Outputs
            % First In, First Out Queue to ensure ongoing outputs
            % Shift elements left (by 4 positions, to make room for 3 new values) and discrads the oldest 3 values
            % Keep at 60 instead of reducing valid to 57 (it would mean to
            % remove the first epoch, but in Calibration it is included so
            % better it is also included in experiment)

            MWL_buffer(1:end-3) = MWL_buffer(4:end);
            MWL_buffer(end-2)   = MWL_stew;                     % Store latest STEW based MWL classification
            MWL_buffer(end-1)   = MWL_heat;                     % Store latest HEAT based MWL classification
            MWL_buffer(end)     = MWL_matb;                     % Store latest MATB based MWL classification
            final_prediction    = MWL_buffer(MWL_buffer >= 0);  % Compute Majority Vote within buffer, remove invalid values (-1)

            valid_count = sum(MWL_buffer >= 0);                 % count up current mwl buffer size ->
                                                                % compare to when buffer is enough filled -> considered as valid:

            if valid_count >= MWL_buffer_valid - 9              % consider as valid: 60 - (3 pred*3 epochs = 9) = 51 predictions before valid predictions start
                final_MWL = mode(final_prediction);             % Get the most frequent value (taking lowest (0) if even)
                major_vote_valid = true;
            else
                final_MWL = NaN;                                % Use current MWL if buffer is empty
                major_vote_valid = false;
            end

            % -------------------------------------------------------------------------
            % Update Live Plot
            % -------------------------------------------------------------------------
            % Skip plotting if figure was closed
            if ~isvalid(h_pred_STEW)
                disp('[WARNING] Live plot closed. Skipping update.');
                continue;  % Skip this iteration
            end

            % Store predictions
            pred_history_STEW(end+1) = MWL_stew;
            pred_history_HEAT(end+1) = MWL_heat;
            pred_history_MATB(end+1) = MWL_matb;
            shadow_history(end+1)    = ground_truth_label;

            % Only update majority vote if buffer has enough valid values
            if major_vote_valid || ep == epochs_per_block
                majority_history(end+1) = final_MWL;           % Plot correct Majority Vote when Buffer is filled completely
            else
                majority_history(end+1) = NaN;                 % Plot Majority Vote at Ground Truth Label
            end

            % Update live plot
            figure(live_mwl_fig);
            set(h_pred_STEW, 'XData', 1:length(pred_history_STEW), 'YData', pred_history_STEW);
            set(h_pred_HEAT, 'XData', 1:length(pred_history_HEAT), 'YData', pred_history_HEAT);
            set(h_pred_MATB, 'XData', 1:length(pred_history_MATB), 'YData', pred_history_MATB);
            set(h_maj_shadow, 'XData', 1:length(shadow_history), 'YData', shadow_history); % take always ground truth until buffer is filled
            set(h_maj, 'XData', 1:length(majority_history), 'YData', majority_history);
            drawnow limitrate;

            % Expand x-axis if needed
            figure(live_mwl_fig);
            xl = xlim(gca);
            if length(pred_history_STEW) > xl(2) - 5
                xlim([length(pred_history_STEW)-90, length(pred_history_STEW)+10]);
                xlim([length(pred_history_HEAT)-90, length(pred_history_HEAT)+10]);
                xlim([length(pred_history_MATB)-90, length(pred_history_MATB)+10]);
            end


            % -------------------------------------------------------------------------
            % 'Adaptive' Robot Update
            % -------------------------------------------------------------------------
            if strcmpi(setting, 'adaptive') && ~already_corrected && ep >= ceil(epochs_per_block - 10) % 30 - 10 = 20 epochs = 40 sec
                if final_MWL == 0
                    correction_command = "LOW";
                    correction_string = ['PREDICTED', correction_command];
                    cue_command = 'HIGH';

                    if strcmpi(state, 'realtime')
                        write(t_py, uint8(cue_command));
                        fprintf('[ADAPT] Adaptation command sent: %s\n', cue_command);
                    else
                        fprintf('[ADAPT] Simulated correction sent: %s\n', cue_command);
                    end

                    % Draw vertical line to mark ADAPT moment
                    figure(live_mwl_fig);
                    xline(epoch_idx, ':', 'Color', [0 0.6 0], 'LineWidth', 2, 'DisplayName', 'off' ,'HandleVisibility', 'off');
                    text(epoch_idx - 2, 0.1, correction_string, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0 0.6 0], ...
                        'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
                    adapt_epoch = epoch_idx;

                    % Show adapted task cue based on final predicted MWL
                    figure(fig_cue);
                    stop(cue_ctrl.timerObj);  % STOP timer before showing message
                    cue_com_msg = sprintf('Adapting Task to %s', cue_command);
                    cue_ctrl.showMessage(cue_com_msg, 2);  % pause 2 sec - show adapting message
                    cue_ctrl.condition = cue_command;
                    cue_ctrl.resetCountdown();
                    cue_ctrl.resetInternalTimer();
                    cue_ctrl.stopFlag = false;
                    cue_ctrl.startTimer();

                else
                    correction_command = "HIGH";
                    correction_string = ['PREDICTED', correction_command];
                    cue_command = 'LOW';

                    if strcmpi(state, 'realtime')
                        write(t_py, uint8(cue_command));
                        fprintf('[ADAPT] Adaptation command sent: %s\n', cue_command);
                    else
                        fprintf('[ADAPT] Simulated correction sent: %s\n', cue_command);
                    end

                    % Draw vertical line to mark ADAPT moment
                    figure(live_mwl_fig);
                    xline(epoch_idx, ':', 'Color', [0 0.6 0], 'LineWidth', 2, 'DisplayName', 'off', 'HandleVisibility', 'off');
                    text(epoch_idx - 2, 0.1, correction_string, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0 0.6 0], ...
                        'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
                    adapt_epoch = epoch_idx;

                    % Show adapted task cue based on final predicted MWL
                    figure(fig_cue);
                    stop(cue_ctrl.timerObj);  % STOP timer before showing message
                    cue_com_msg = sprintf('Adapting Task to %s', cue_command);
                    cue_ctrl.showMessage(cue_com_msg, 2);  % pause 2 sec - show adapting message
                    cue_ctrl.condition = cue_command;
                    cue_ctrl.resetCountdown();
                    cue_ctrl.resetInternalTimer();
                    cue_ctrl.stopFlag = false;
                    cue_ctrl.startTimer();

                end

                already_corrected = true;
            end


            % -------------------------------------------------------------------------
            % Logging
            % -------------------------------------------------------------------------
            % Calculate Epoch End Time relative to Experiment Start
            epoch_end_time = toc(experiment_start_time);

            % Save Task Log
            experiment_log(epoch_idx).true_label         = ground_truth_label;    % Store Current Ground Truth (from Robot Task)
            experiment_log(epoch_idx).predicted_MWL_stew = MWL_stew;              % Individual Epoch MWL prediction
            experiment_log(epoch_idx).predicted_MWL_heat = MWL_heat;              % Individual Epoch MWL prediction
            experiment_log(epoch_idx).predicted_MWL_matb = MWL_matb;              % Individual Epoch MWL prediction
            experiment_log(epoch_idx).majority_MWL       = final_MWL;             % Majority of FIFO Buffer MWL prediction

            % Save Correction Command
            if exist('correction_command', 'var')
                experiment_log(epoch_idx).adapt_command = correction_command;     % Save the Block Level Single Correction Command
            else
                experiment_log(epoch_idx).adapt_command = "";
            end

            % Compute Accuracy Dynamically/ Live
            if ~isnan(final_MWL) && ~already_corrected
                experiment_log(epoch_idx).correct = (final_MWL == ground_truth_label);
            else
                experiment_log(epoch_idx).correct = NaN;
            end

            % Mark Adapted Epochs
            if strcmpi(setting, 'adaptive') && already_corrected
                experiment_log(epoch_idx).adapted_epochs = adapt_epoch;           % Mark Epochs from where Adaptation was performed
                adapt_epoch = adapt_epoch + 1;
            else
                experiment_log(epoch_idx).adapted_epochs = NaN;
            end

            experiment_log(epoch_idx).STEW_classifier_confidence = score_stew;    % Store STEW Classifier Confidence Values for later Analysis
            experiment_log(epoch_idx).HEAT_classifier_confidence = score_heat;    % Store HEAT Classifier Confidence Values for later Analysis
            experiment_log(epoch_idx).MATB_classifier_confidence = score_matb;    % Store MATB Classifier Confidence Values for later Analysis

            % Log Time and Data
            experiment_log(epoch_idx).epoch_start   = epoch_start_time;           % Epoch Start Time
            experiment_log(epoch_idx).epoch_end     = epoch_end_time;             % Epoch End Time
            experiment_log(epoch_idx).full_raw      = raw_pre_sample_rt;          % Truly Raw EEG without resampling/ downsampling (250 Hz)
            experiment_log(epoch_idx).raw           = epoch_data_acq;             % Save Raw Epoch Data from Nautilus (128Hz)
            experiment_log(epoch_idx).raw_ordered   = epoch_data;                 % Save Raw Epoch Data Reordered in Emotiv Epoch Ch-Setup
            experiment_log(epoch_idx).processed     = processed_epoch;            % Save Processed Epoch Data
            experiment_log(epoch_idx).STEW_features = epoch_features_STEW;        % Save STEW Epoch Features
            experiment_log(epoch_idx).HEAT_features = epoch_features_HEAT;        % Save HEAT Epoch Features
            experiment_log(epoch_idx).MATB_features = epoch_features_MATB;        % Save STEW Epoch Features

            fprintf('Epoch %3d | Ground Truth: %d | MWL (STEW): %d | MWL (HEAT): %d | MWL (MATB): %d | MAJORITY VOTE MWL: %d \n', ...
                epoch_idx, ground_truth_label, MWL_stew, MWL_heat, MWL_matb, final_MWL);

            % Increment Epoch Counter
            epoch_idx = epoch_idx + 1;

            % Clear Command
            correction_command = "";

            % Add 2 sec delay to simulate real time
            % pause(2);

        end  % Finishes 1 Block of either Low or High MWL Robot Task Finishes ->
        % Within this it finishes: All related EEG Epochs (buffering/ processing/ feature extraction/ prediction/ logging)

        % Stop robot or simulate rest
        if strcmpi(state, 'realtime')
            write(t_py, uint8(command_stop));
            fprintf('[STOP] Block %d done. Robot Stopped.\n', rt_block_idx);

            % fprintf('[PAUSE] 15sec Pause for Resting.\n');
            % pause(15);  % Resting for 15 sec
        else
            fprintf('[TEST] Block %d done. Simulated robot command STOP.\n', rt_block_idx);
            % fprintf('[PAUSE] 1sec Pause for Resting.\n');
            % pause(1);  % Resting for 1 sec (testing)
        end

        % Stop Cue Screen
        cue_ctrl.stopFlag = true;
        stop(cue_ctrl.timerObj);

        figure(fig_cue);
        block_done_msg = sprintf('Block %d / %d Done. \n___________________\n \nWaiting for Confirmation \nto start next Block.', rt_block_idx, num_blocks);
        cue_ctrl.showMessage(block_done_msg, 0); % pause - Show adapting message

        % Enter "REDO" to repeat Block or "STOP" to end the Real-Time Loop or press "Enter" to continue
        end_experiment_input = input('Press ENTER to continue, type "REDO" to repeat Block, or "STOP" to end: ', 's');   % Waits for User Input
        if strcmpi(end_experiment_input, 'STOP')
            write(t_py, uint8(command_stop));
            fprintf('[STOP] Experiment manually stopped by user.\n');
            return;                                                              % Exerpiment Permanently Interrupted -> Exit Loop
        elseif strcmpi(end_experiment_input, 'REDO')
            fprintf('[REDO] Repeating current Block %d...\n', rt_block_idx);

            % Remove last block's data (30 epochs)
            epoch_idx = epoch_idx - epochs_per_block;
            experiment_log(epoch_idx + 1:end) = [];

            % Also remove history and live plot data
            pred_history_STEW(end - epochs_per_block + 1:end) = [];
            pred_history_HEAT(end - epochs_per_block + 1:end) = [];
            pred_history_MATB(end - epochs_per_block + 1:end) = [];
            majority_history(end - epochs_per_block + 1:end)  = [];
            shadow_history(end - epochs_per_block + 1:end)    = [];

            continue;  % Re-run same block
        else
            break;
        end
    end

    % Reset the MWL Buffer to refill it from empty state for new block
    MWL_buffer = -1 * ones(MWL_buffer_valid, 1);

end % Finishes all Blocks of Experiment

% Experiment Finished - Stop Robot
write(t_py, uint8(command_stop));

% End Screen
figure(fig_cue);
block_done_msg = sprintf('Experiment Finished!\nThank you for participating!');
cue_ctrl.showMessage(block_done_msg, 0);  % pause - Show adapting message

% Save Experiment Results
fprintf('\n[INFO] Experiment Complete! Saving Data...\n');
save_name = sprintf('Subject%d_Results.mat', subject_number);
save(save_name, 'experiment_log', 'experiment_log_metadata');

% Save Live Plot
figure(live_mwl_fig);
if isvalid(h_pred_STEW)  % Only try saving if the plot still exists
    filename_liveplot = sprintf('Subject%d_LivePlot', subject_number);
    save_plot_all_formats(gcf, filename_liveplot);
end


% -------------------------------------------------------------------------
% Save Summary Log (.txt)
% -------------------------------------------------------------------------
summary_filename = sprintf('Subject%d_Results_Summary.txt', subject_number);
fid = fopen(summary_filename, 'w');

% Get Total Epoch Time Duration
epoch_starts = [experiment_log.epoch_start];
epoch_ends = [experiment_log.epoch_end];
epoch_durations = epoch_ends - epoch_starts;
epoch_durations_first = epoch_durations(2:end);        % Remove the First Epoch (Mainly for Offline Testing, bc there is
% delay for single time simulation data loading)
avg_duration = mean(epoch_durations_first);            % Avg durations

fprintf(fid, '========= Real-Time pBCI Experiment Summary =========\n');
fprintf(fid, '\n---------------- General Settings -----------------\n');
fprintf(fid, 'Subject ID        : %d\n', subject_number);
fprintf(fid, 'Date & Time       : %s\n', datetime("now"));
fprintf(fid, 'Model 1           : %s\n', mdl_name_stew);
fprintf(fid, 'Model 2           : %s\n', mdl_name_heat);
fprintf(fid, 'Model 3           : %s\n', mdl_name_matb);
fprintf(fid, '# Channels        : %d\n', nbchan);
fprintf(fid, 'Selected Channels : %s\n', strjoin(channels, ', '));
fprintf(fid, 'Epoch Length      : %.1f sec\n', epoch_sec);
fprintf(fid, 'Sampling Rate     : %d Hz\n', fs);
fprintf(fid, 'Overlap           : %.1f %%\n', overlap*100);
fprintf(fid, 'Total Epochs      : %d\n', length(experiment_log));
fprintf(fid, 'Avg Process Time  : %.4f sec (Loading Epoch until after ADAPT Command)\n', avg_duration);
realtime_label_string = sprintf('%d', sequence);
fprintf(fid, 'Block Label Order : %s\n', realtime_label_string);


% Only look at predictions vs true labels before applying "ADAPT"
pred_idxs = isnan([experiment_log.adapted_epochs]);

fprintf(fid, '\nAll metrics are computed on Pre-ADAPT epochs only.\n');
fprintf(fid, 'Pre-ADAPT Epochs Evaluated: %d\n', sum(pred_idxs));

% -------------------------------------------------------------------------
fprintf(fid, '\n---------------------- STEW ----------------------');
fprintf(fid, '\n-------------- Classwise Accuracies --------------\n');

final_preds_stew = [experiment_log(pred_idxs).predicted_MWL_stew];
true_labels_stew = [experiment_log(pred_idxs).true_label];

% Confusion matrix
[confmat_stew, ~] = confusionmat(true_labels_stew, final_preds_stew);

TN = confmat_stew(1,1);  % True negatives (LOW)
FP = confmat_stew(1,2);  % False positives (predicted HIGH, actually LOW)
FN = confmat_stew(2,1);  % False negatives (predicted LOW, actually HIGH)
TP = confmat_stew(2,2);  % True positives (HIGH)

% Basic counts
fprintf(fid, 'Low MWL Count     : %d (Ground Truth), %d (Predicted)', ...
    sum(true_labels_stew==0), sum(final_preds_stew==0));
fprintf(fid, '\n');
fprintf(fid, 'High MWL Count    : %d (Ground Truth), %d (Predicted)\n', ...
    sum(true_labels_stew==1), sum(final_preds_stew==1));
fprintf(fid, '\n');

% Metrics for HIGH MWL (class = 1)
precision_high_stew = TP / (TP + FP);
recall_high_stew    = TP / (TP + FN);
f1_score_high_stew        = 2 * (precision_high_stew * recall_high_stew) / (precision_high_stew + recall_high_stew);

fprintf(fid, 'Precision (High)  : %.2f %%\n', 100 * precision_high_stew);
fprintf(fid, 'Recall (High)     : %.2f %%\n', 100 * recall_high_stew);
fprintf(fid, 'F1-Score (High)   : %.2f %%\n', 100 * f1_score_high_stew);

% Accuracy
acc_stew = 100 * (TP + TN) / sum(confmat_stew(:));
fprintf(fid, '\nSTEW Accuracy     : %.2f %%\n', acc_stew);

% Plot STEW confusion matric
fig_stew = figure;
chart_stew = confusionchart(confmat_stew, {'Low MWL', 'High MWL'});
chart_stew.Title = sprintf('Confusion Matrix - STEW - Subject %d', subject_number);
chart_stew.RowSummary = 'row-normalized';       % Shows percentages per actual class
chart_stew.ColumnSummary = 'column-normalized'; % (optional) shows percentages per predicted class
chart_stew.Normalization = 'absolute';          % Keeps absolute values in the chart
save_plot_all_formats(fig_stew, sprintf('Subject%d_ConfusionMatrix_STEW', subject_number));


% -------------------------------------------------------------------------
fprintf(fid, '\n---------------------- HEAT ----------------------');
fprintf(fid, '\n-------------- Classwise Accuracies --------------\n');

final_preds_heat = [experiment_log(pred_idxs).predicted_MWL_heat];
true_labels_heat = [experiment_log(pred_idxs).true_label];

% Confusion matrix
[confmat_heat, ~] = confusionmat(true_labels_heat, final_preds_heat);

TN = confmat_heat(1,1);
FP = confmat_heat(1,2);
FN = confmat_heat(2,1);
TP = confmat_heat(2,2);

fprintf(fid, 'Low MWL Count     : %d (Ground Truth), %d (Predicted)', ...
    sum(true_labels_heat==0), sum(final_preds_heat==0));
fprintf(fid, '\n');
fprintf(fid, 'High MWL Count    : %d (Ground Truth), %d (Predicted)\n', ...
    sum(true_labels_heat==1), sum(final_preds_heat==1));
fprintf(fid, '\n');

% Metrics for HIGH MWL (class = 1)
precision_high_heat = TP / (TP + FP);
recall_high_heat    = TP / (TP + FN);
f1_score_high_heat  = 2 * (precision_high_heat * recall_high_heat) / ...
                      (precision_high_heat + recall_high_heat);

fprintf(fid, 'Precision (High)  : %.2f %%\n', 100 * precision_high_heat);
fprintf(fid, 'Recall (High)     : %.2f %%\n', 100 * recall_high_heat);
fprintf(fid, 'F1-Score (High)   : %.2f %%\n', 100 * f1_score_high_heat);

% Accuracy
acc_heat = 100 * (TP + TN) / sum(confmat_heat(:));
fprintf(fid, '\nHEAT Accuracy     : %.2f %%\n', acc_heat);

% Plot HEAT confusion matrix
fig_heat = figure;
chart_heat = confusionchart(confmat_heat, {'Low MWL', 'High MWL'});
chart_heat.Title = sprintf('Confusion Matrix - HEAT - Subject %d', subject_number);
chart_heat.RowSummary = 'row-normalized';  
chart_heat.ColumnSummary = 'column-normalized'; 
chart_heat.Normalization = 'absolute';         
save_plot_all_formats(fig_heat, sprintf('Subject%d_ConfusionMatrix_HEAT', subject_number));



% -------------------------------------------------------------------------
fprintf(fid, '\n---------------------- MATB ----------------------');
fprintf(fid, '\n-------------- Classwise Accuracies --------------\n');

final_preds_matb = [experiment_log(pred_idxs).predicted_MWL_matb];
true_labels_matb = [experiment_log(pred_idxs).true_label];

% Confusion matrix
[confmat_matb, ~] = confusionmat(true_labels_matb, final_preds_matb);

TN = confmat_matb(1,1);
FP = confmat_matb(1,2);
FN = confmat_matb(2,1);
TP = confmat_matb(2,2);

fprintf(fid, 'Low MWL Count     : %d (Ground Truth), %d (Predicted)', ...
    sum(true_labels_matb==0), sum(final_preds_matb==0));
fprintf(fid, '\n');
fprintf(fid, 'High MWL Count    : %d (Ground Truth), %d (Predicted)\n', ...
    sum(true_labels_matb==1), sum(final_preds_matb==1));
fprintf(fid, '\n');

% Metrics for HIGH MWL (class = 1)
precision_high_matb = TP / (TP + FP);
recall_high_matb    = TP / (TP + FN);
f1_score_high_matb  = 2 * (precision_high_matb * recall_high_matb) / ...
                      (precision_high_matb + recall_high_matb);

fprintf(fid, 'Precision (High)  : %.2f %%\n', 100 * precision_high_matb);
fprintf(fid, 'Recall (High)     : %.2f %%\n', 100 * recall_high_matb);
fprintf(fid, 'F1-Score (High)   : %.2f %%\n', 100 * f1_score_high_matb);

% Accuracy
acc_matb = 100 * (TP + TN) / sum(confmat_matb(:));
fprintf(fid, '\nMATB Accuracy     : %.2f %%\n', acc_matb);

% Plot MATB confusion matrix
fig_matb = figure;
chart_matb = confusionchart(confmat_matb, {'Low MWL', 'High MWL'});
chart_matb.Title = sprintf('Confusion Matrix - MATB - Subject %d', subject_number);
chart_matb.RowSummary = 'row-normalized'; 
chart_matb.ColumnSummary = 'column-normalized'; 
chart_matb.Normalization = 'absolute';        
save_plot_all_formats(fig_matb, sprintf('Subject%d_ConfusionMatrix_MATB', subject_number));



% Save Majority Prediction Accuracy
fprintf(fid, '\n--------------- GLOBAL PERFORMANCE ---------------\n');

% Save Global Prediction counts
fprintf(fid, '----------- Global Classwise Accuracies ----------\n');
combined_precision_high = mean([precision_high_stew, precision_high_matb, precision_high_heat]);
combined_recall_high    = mean([recall_high_stew, recall_high_matb, recall_high_heat]);
combined_f1_score_high  = mean([f1_score_high_stew, f1_score_high_matb, f1_score_high_heat]);

% Metrics High
fprintf(fid, 'Global Precision (High)  : %.2f %%\n', 100 * combined_precision_high);
fprintf(fid, 'Global Recall (High)     : %.2f %%\n', 100 * combined_recall_high);
fprintf(fid, 'Global F1-Score (High)   : %.2f %%\n', 100 * combined_f1_score_high);


fprintf(fid, '\n---------------- GLOBAL ACCURACY -----------------\n');
combined_accuracy = mean([acc_stew, acc_heat, acc_matb]);
fprintf(fid, 'Total Combined Accuracy  : %.2f %%\n', combined_accuracy);
fprintf(fid, '--------------------------------------------------\n');

fprintf(fid, '\n------------- MAJORITY VOTE ACCURACY -------------\n');

% Compute Total Valid Majority Vote Buffer Accuracy (valid + the moment the
% adapt command is sent
% Where mwl is not NaN in .correct: 
majority_correct = ~isnan([experiment_log.correct]);
valid_majority_correct = find(majority_correct);        % Get the positions/ epoch idxs

% including the one where the adaptation is sent
all_corrections = cellfun(@(x) x, {experiment_log.adapt_command});
adapt_idxs = find(strcmp(all_corrections, "HIGH") | strcmp(all_corrections, "LOW"));           

% Get all correct epoch idxs or the adapt_idxs
combined_idxs = sort([valid_majority_correct, adapt_idxs]);

% Get majority vote buffer predictions
majority_preds = [experiment_log(combined_idxs).majority_MWL];

% Get corresponding ground truth labels
true_labels = [experiment_log(combined_idxs).true_label];

% Compute Accuracy
total_majority_buffer_accuracy = 100 * mean(majority_preds == true_labels);
fprintf(fid, 'Majority MWL Buffer Accuracy       : %.2f %%\n', total_majority_buffer_accuracy);

% Compute BLOCK LEVEL accuracy x / numBlocks
% Extract all non-empty correction commands
all_corrections = {experiment_log.adapt_command};
adapt_idxs = find(~cellfun(@isempty, all_corrections));   % Indices where ADAPT command was sent
correction_cmds = all_corrections(adapt_idxs);            % Extract the commands in order

% Convert commands to binary values
predicted_block_labels = zeros(1, length(correction_cmds));
for i = 1:length(correction_cmds)
    if strcmpi(correction_cmds{i}, 'HIGH')
        predicted_block_labels(i) = 1;
    elseif strcmpi(correction_cmds{i}, 'LOW')
        predicted_block_labels(i) = 0;
    else
        predicted_block_labels(i) = NaN;
    end
end

% Remove NaNs from predicted_block_labels and match indices in sequence
valid_idx = ~isnan(predicted_block_labels);
predicted_block_labels = predicted_block_labels(valid_idx);

% Ground truth block labels from sequence
true_block_labels = sequence(1:length(predicted_block_labels));  % make sure same length

% Compute block-level accuracy
block_level_accuracy = mean(predicted_block_labels == true_block_labels) * 100;
fprintf(fid, 'ADAPT Command Block-Level Accuracy : %.2f %%\n', block_level_accuracy);

fprintf(fid, '--------------------------------------------------\n');
fclose(fid);

fprintf('\n__________________________________\n');
fprintf('\n[DONE] Finished Experiment.\n\n');


