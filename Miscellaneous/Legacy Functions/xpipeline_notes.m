%%% Processing Pipeline

% "Real-Time Adaptive pBCI Robot Behavior Modification"

%% Structure

% 1. Get EEG data, including a buffer from live recording (remove and read
% the mean of n samples) / % 1. Get EEG data and Epoching EEG into 2 second segments (maybe 50% overlap) 
% 2. Referencing (e.g., Average Referencing to Mastoid Electordes)
% 3. Filtering (0.5 - 40 Hz Bandpass Filtering)
% 4. Artifact Removal (ASR) (EEGLAB Toolbox)
% 5. Features Extraction (FFT (mainly: theta, alpha, beta), Engagement Index (e.g., beta / alpha + theta), Functional Connectivity using Coherence (and maybe Phase-Locking)
% 5.5 (For Experiment) Calibration of pre-trained Classifier for individuals (using Transfer Learning) (Deep Learning Toolbox)
% 6. Classification - 2 options: 
% 6.1. 1 classifier, but Ensemble approach ("Stacked Classifier") where we combine several ML approaches to draw a final classification from
% 6.2. 2 classifiers, 1 cognitive (MWL) & 1 affective (Stress/Relaxed) 
% 7. Classifier Output -> adaptive Robot Instructions 

%% 1. Get EEG data

% Reading and listing all Files with 'EEG'
eeg_data = ls('EEG*');
% optionally:
eeg_data = load('eeg_data.mat'); % always update the current file name during experiment

% Data should be in format: Channels x Time (x Trials)

% Sampling Rate
fs = 128; % Sampling Frequency 128 Hz

% Buffer Setup
buffer_length = 2 * fs; 
EEG_buffer = zeros(nbchan, buffer_length);         % initialize the matrix to store the eeg data within the buffer length
overlap = 0.5;                                     % adding a 50% overlap
step_size = round((1-overlap) * buffer_length);    % step of 0.5 the length of the buffer between the whole buffer sequence

while true
    continous_eeg = acquire_new_data();                                  % some EEG stream function/ way to get the new continous eeg
    EEG_buffer = circshift(EEG_buffer, [0, -size(continous_eeg, 2)]);    % shift buffer to the left to add new data at the end
    EEG_buffer(:, end-size(continous_eeg,2)+1 : end) = continous_eeg;    % add new continous eeg data

    % Process latest 2 seconds of data with overlap
    for start_idx = 1:step_size:(size(EEG_buffer,2) - buffer_length +1)
        eeg_data = EEG_buffer(:,start_idx:start_idx + buffer_length -1);
    
    % ... all the next processing steps as well as feature extraction and
    % classification/ prediction and the robot instructions would be within
    % this while loop
    end
end

%%ChatGPT Version: 
%%Initialize Buffer
% buffer_length = 2 * fs; 
% EEG_buffer = zeros(nbchan, buffer_length);  
% coherence_overlap = 0.5; 
% step_size = round((1 - coherence_overlap) * buffer_length);
% 
% % Circular Buffer Alternative
% bufferIndex = 1; 
% 
% while true
%     continous_eeg = acquire_new_data();  % Replace with actual EEG stream function
%     
%     % Circular buffer implementation (better memory efficiency)
%     EEG_buffer(:, bufferIndex:bufferIndex + size(continous_eeg, 2) - 1) = continous_eeg;
%     bufferIndex = mod(bufferIndex + size(continous_eeg, 2), buffer_length) + 1;
%     
%     % Process overlapping segments
%     for start_idx = 1:step_size:(size(EEG_buffer,2) - buffer_length +1)
%         eeg_data_segment = EEG_buffer(:, start_idx:start_idx + buffer_length -1);
%         % Continue processing this segment
%     end
% end

%% 2. Referecing

% Referencing to mastoid (or any other) channel
%ref = mean(eeg_data.mastoid,2);         % check which one is the eeg_data.mastoid electrode/ channel and then replace it here accordingly
%eeg_data_referenced = eeg_data - ref; 

% Other option: Average Referencing (by subtracting the mean of all
% channels)
ref = mean(eeg_data,1); % Average referencing
eeg_data_referenced = eeg_data - ref;


%% 3. Filter

low_cutoff = 1;
high_cutoff = 40;

% More high-level approach using bandpass function, less control, more risk
% of introducing phase distortion - but quick and straight forward
wpass = [low_cutoff high_cutoff];
eeg_data_filtered = bandpass(eeg_data_referenced, wpass, fs);

% More flexibility (control over order, type, etc.) approach using butter filter applied with filtfilt 
% ensures zero-phase filtering (eliminate phase distortion introduced
% through filtering by applying both "Forward Filtering" and "Reverse
% Filtering" - reversing the phase shifts introduced in the first step)
[b,a] = butter(4, wpass / (fs/2), 'bandpass');          % 4th order Butterworth filter, normalized by fs/2 , getting filter coefficients numerator b and denominator a
eeg_data_filtered = filtfilt(b,a,eeg_data_referenced);  % applying filter to referenced EEg data 

%% 4. Artifact Removal using Artifact Subspace Removal (ASR) 
% using the EEGLAB Toolbox

% Load data into EEGLAB format
eeg_data_filtered = pop_importdata('dataformat', 'matlab', 'data', eeg_data_filtered, 'srate', fs);

% Preparing the data to process with ASR: specifying the channels to which
% the cleanline algorithm will be applied; normalizing the spectrum of the
% EEG data (can help improve the effectiveness of noise removal process)
% Also contains a 2Hz Bandwith for the Line Noise Filter
% pop_cleanline is only necessary if i am sure that i filtered the signal
% sufficiently beforehand. 
% eeg_data_filtered = pop_cleanline(eeg_data_filtered, 'bandwidth',2, 'chanlist',1:eeg_data_filtered.nbchan, 'linefreqs', [50 100], 'normSpectrum', 1);

% for ASR is important to have a >1Hz High-Pass Filtered Signal (this is
% done in the Filter Section by Filtering between 1 and 40 Hz.
% Apply ASR
eeg_data_asr = pop_runasr(eeg_data_filtered, 'threshold', 3.5); % Threshold specifies how aggressively ASR is filtering (3-4 is recommended, 5 already very aggressive)


%% 5. Epoching
% Dividing the EEG signal into segments of 2 seconds for offline processing (e.g., for the training data)

epoch_length_sec = 2;
epoch_length = epoch_length_sec * fs;               % 2 seconds
overlap = 0.5;                                      % adding a 50% overlap
step_size = round((1-overlap) * epoch_length);      % step of 0.5 the length of the buffer between the whole buffer sequence
num_epochs = floor(length(eeg_data_asr,2) - epoch_length / step_size + 1);

% Initialize epochs matrix
epochs = zeros(nbchan, epoch_length, num_epochs);

% first approach: (but "reshape" assumes perfect alignment, which can be a
% problem with real EEG data)
%epochs = reshape(eeg_data_asr(1:num_epochs*epoch_length), epoch_length, num_epochs);    % shapes matrix to epoch length x number of epochs

% better approach:

% Generate Epochs
for i = 1:num_epochs
    start_idx = (i-1)*step_size+1;
    end_idx = start_idx + epoch_length -1;
    epochs(:,:,i) = eeg_data(:,start_idx:end_idx);
end

% for online processing (e.g., for continous eeg data stream from live
% recordings) i will use a different approach (the initialization in
% general will be different) - the EEG data processed here will already be
% segemented in 2 sec intervals coming from the buffer.

%% 6. Feature Extraction

% Frequency Bands
theta_band = [4 8];  % Theta
alpha_band = [8 12]; % Alpha
beta_band = [13 30]; % Beta

% Initialize Feature Matrix
features = zeros(num_epochs,7)  % 3 Columns for theta, alpha and beta features + 1 more for functional connectivity (more can be added)
                                % e.g., adding "Mobility", "Complexity", "Spectrum Entropy"

for i = 1:num_epochs

    % Extract single epoch data
    epoch_data = squeeze(epochs(:,:,i));    % Extracts 2D matrix (Channels x Samples)

    % Compute FFT
    epoch_fft = fft(epoch_data,[],2);                   % FFT along time dimension
    fft_spectrum = abs(epoch_fft/epoch_length);         % full two sided frequency spectrum
    fft_spectrum = fft_spectrum(1:epoch_length/2);      % single sided
    fft_spectrum(2:end-1) = 2*fft_spectrum(2:end-1);    % Amplitude Correction

    % Frequency Vector
    f = linspace(0, fs/2, epoch_length/2);

    % f = fs*(0:(epoch_length/2)) / epoch_length;
    % Creates a fq vector that includes nyquist fq and is based on FFT
    % output. "0:(epoch_length/2)" generates indicies that correspond to fq
    % bin of the FFT. Resulting vector will have "epoch_length/2 +1) pnts.
    % (for single sided fft spectrum)

    % Extract power in fq bands (should i extract the mean of the
    % bandpowers here?) "mean((...),'all')"
    features(i,1) = bandpower(fft_spectrum,f,theta_band,'psd');   % Theta Power
    features(i,2) = bandpower(fft_spectrum,f,alpha_band,'psd');   % Alpha Power
    features(i,3) = bandpower(fft_spectrum,f,beta_band,'psd');    % Beta Power

    % Compute Functional Connectivity (Coherence)
    % Coherence is a measure of degree of correlation between 2 signals at
    % different frequencies. Useful in EEG analysis to assess functional
    % connectivity between brain regions (electrodes)

    % Window for Coherence Calculation
    window_length = 256;
    coherence_overlap = window_length / 2; % 50% overlap

    % nbchan = size(eeg_data,2);
    coherence_matrix = zeros(nbchan, nbchan, num_epochs);    % initialize connectivity matrix to store coherence values for channel pairs

    for j = 1:nbchan
        for k = j+1:nbchan

            % Compute the magnitude-squared coherence between 2 signals (epochs)
            [Cxy, F] = mscohere(epoch_data(j,:), epoch_data(k,:), hamming(window_length), coherence_overlap, window_length, fs);

            coherence_matrix(j,k) = mean(Cxy);               % Average Coherence Values over frequency range to create coherence matrix for each epoch
            coherence_matrix(k,j) = coherence_matrix(j,k);   % Symmetric Matrix

        end
    end

    % Add the coherence features for each epoch
    features(i,4) = mean(coherence_matrix(:));   % Average coherence across all pairs for epoch i

    % Other Features that could be implemented: 
    % Hjorth Parameters (Mobility, Complexity)
    diff1 = diff(epoch_data,1,2);   % First derivative
    diff2 = diff(diff1,1,2);        % Second derivative

    mobility = std(diff1,0,2) ./ std(epoch_data,0,2);   % Hjorth Mobility
    complexity = std(diff2,0,2) ./ std(diff2,0,2);      % Hjorth Complexity

    % Entropy Measures
    normalized_spectrum = fft_spectrum ./ sum(fft_spectrum,2)                               % Normalized Spectrum
    spectrum_entropy = -sum(normalized_spectrum .* log(normalized_spectrum),2,'omitnan');   % Shannon Spectrum

    % Adding new features to features vector
    features(:,5) = mean(mobility);
    features(:,6) = mean(complexity);
    features(:,7) = mean(spectrum_entropy);
    
end


%% 7.5 Transfer Learning

% Use Matlab Deep Learning Toolbox to implement Transfer
% Learning to allow to calibrate the existing pre-trained model based on
% new data.

%% 8. Classification

% Get labels matrix from training dataset.

%% 1. Single Classifier, SVM approach

% Train Classifier
mdl_svm = fitcsvm(features, labels);

% Predictions
predictions_svm = predict(mdl_svm, features);



%% 2. Stacked Classifier, Ensemble Approach (Combining Decision Tree, KNN, SVM classifier outputs)

% Train Classifier
mdl_esemble = fitcensemble(features, labels, 'Method', 'Bag');

% Predictions
predictions_ensemble = predict(mdl_ensemble, features);

%% 3. Multimodal Classifier (Classifying High/Low Mental Workload Level + High/Low Stress Level)

% Assuming i have features and labels for both classifiers
% features_workload: Features for Mental Workload classification
% labels_workload: Labels for Mental Workload (1 for High, 0 for Low)
% features_stress: Features for Stress classification
% labels_stress: Labels for Stress (1 for High, 0 for Low)

% Train the Mental Workload Classifier
mdl_workload = fitcsvm(features_workload, labels_workload); % SVM for Mental Workload

% Train the Stress Level Classifier
mdl_stress = fitcsvm(features_stress, labels_stress); % SVM for Stress Level

% Assuming you have new data for prediction
% new_data: New features for prediction (same structure as training data)

% Predict Mental Workload Level
predicted_workload = predict(mdl_workload, new_data);

% Predict Stress Level
predicted_stress = predict(mdl_stress, new_data);

% Combine Predictions
% Initialize action variable
action = '';

for i = 1:length(predicted_workload)
    if predicted_workload(i) == 1 && predicted_stress(i) == 1
        action = 'Decrease Velocity'; % High Mental Workload and High Stress
    elseif predicted_workload(i) == 1 && predicted_stress(i) == 0
        action = 'Decrease Velocity'; % High Mental Workload and Low Stress
    elseif predicted_workload(i) == 0 && predicted_stress(i) == 1
        action = 'Decrease Velocity'; % Low Mental Workload and High Stress
    elseif predicted_workload(i) == 0 && predicted_stress(i) == 0
        action = 'Increase Velocity'; % Low Mental Workload and Low Stress
    end
    
    % Output the action for the current prediction
    fprintf('Prediction %d: %s\n', i, action);
end


%% Calibration

% Selecting data from the train and cross-dataset to test for calibration

% Get Train Data
[~, ~, train_features, val_features, test_features, ~, val_labels, test_labels] = get_data('HEATCHAIR');

% Get Calibration Data
[cross_features, cross_labels] = get_data('MATB_easy_meddiff');

% Select Calibration Subset (as per RT Pipeline Calibration Phase)
% Specify Calibration Samples Needed:
params.samples = 360;
samples_per_class = params.samples / 2; % Half of Calibration Samples (for each class one half)
train_ratio = 0.7;
total_required_samples = round(params.samples / train_ratio);
samples_per_class_total = round(total_required_samples / 2);
val_test_samples_per_class = round((samples_per_class_total - samples_per_class)/2);

% Take small Sample Size of the Cross-Data to use for fine-tuning
low_idx = find(cross_labels == 0);
high_idx = find(cross_labels == 1);
rng(42);
low_idx = low_idx(randperm(length(low_idx)));
high_idx = high_idx(randperm(length(high_idx)));

% Split 70% / 15% / 15%
train_idx = [low_idx(1:samples_per_class); high_idx(1:samples_per_class)];
val_idx = [low_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class); high_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class)];
test_idx = [low_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total); high_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total)];
train_idx = train_idx(randperm(length(train_idx)));
val_idx = val_idx(randperm(length(val_idx)));
test_idx = test_idx(randperm(length(test_idx)));

% Training Data
tuning_train_features = cross_features(train_idx,:);
tuning_train_labels = cross_labels(train_idx);

% Validation and Testing Data (remaining unseen data)
tuning_val_features = cross_features(val_idx,:);
tuning_val_labels = cross_labels(val_idx);

% For now not using the test data - will be used if further 
% Hyperparameter Tuning included
tuning_test_features = cross_features(test_idx,:);
tuning_test_labels = cross_labels(test_idx);

% Display structure
fprintf('\n=== Final Stratified Split (Fully Balanced) ===\n');
fprintf('Train: %d samples |  Low: %d | High: %d\n', ...
    length(tuning_train_labels), sum(tuning_train_labels==0), sum(tuning_train_labels==1));
fprintf('Val:   %d samples |  Low: %d | High: %d\n', ...
    length(tuning_val_labels), sum(tuning_val_labels==0), sum(tuning_val_labels==1));
fprintf('Test:  %d samples |  Low: %d | High: %d\n', ...
    length(tuning_test_labels), sum(tuning_test_labels==0), sum(tuning_test_labels==1));





% -------------------------------------------------------------------------
% Domain Adapation
% -------------------------------------------------------------------------
% Feature Normalization: Transform the new incoming data in a way that it
% fits into the Normalization of the Training Data.

if only_domain_adaptation

    fprintf('Performing Domain Adaptation ...\n');

    % Load Normalized and Hyperparameter Tuned Normalized Model
    norm_mdl_workload_loader = load('24_4sec_proc3wRef_HEATCHAIR_norm_model.mat');
    norm_mdl_workload = norm_mdl_workload_loader.norm_mdl_workload;

    hyper_norm_mdl_workload_loader = load('hyper_24_4sec_proc3wRef_HEATCHAIR_norm_model.mat');
    hyper_norm_mdl_workload = norm_mdl_workload_loader.norm_mdl_workload;

    % Get the Statistics for the Normalization from the Training Data
    mu = mean(train_features);
    sigma = std(train_features);
    sigma(sigma == 0) = 1;       % Avoid division by zero (if any feature has zero variance)
    
    % Perform Z-score Normalization to all Features of the Training and
    % Cross Data

    % Normalize the Training Data
    norm_train_features = (train_features - mu) ./ sigma;
    norm_val_features = (val_features - mu) ./ sigma;
    norm_test_features = (test_features - mu) ./ sigma;

    % Normalize Cross-Data Features based on Training Data
    norm_adapted_train_features = (tuning_train_features - mu) ./ sigma;
    norm_adapted_val_features = (tuning_val_features - mu) ./ sigma;
    norm_adapted_test_features = (tuning_test_features - mu) ./ sigma;

    % Evaluate Model on Cross Data After Domain Adaptation 
    eval_mdl_performance(norm_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], 'Normalized Model after Domain Adaptation - Cross Data');

    % Evaluate Hyperparameter Tuned Model on Cross Data After Domain Adaptation 
    eval_mdl_performance(hyper_norm_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], 'Hyperparameter Tuned Normalized Model after Domain Adaptation - Cross Data');

    fprintf('Domain Adaptation Evaluation Complete.\n');

elseif do_domain_adaptation

    fprintf('Performing Domain Adaptation for further Transfer Learning...\n');

    % Get the Statistics for the Normalization from the Training Data
    mu = mean(train_features);
    sigma = std(train_features);
    sigma(sigma == 0) = 1;       % Avoid division by zero (if any feature has zero variance)

    % Perform Z-score Normalization to all Features of the Training and
    % Cross Data

    % Normalize the Training Data
    norm_train_features = (train_features - mu) ./ sigma;
    norm_val_features = (val_features - mu) ./ sigma;
    norm_test_features = (test_features - mu) ./ sigma;

    % Normalize Cross-Data Features based on Training Data
    norm_adapted_train_features = (tuning_train_features - mu) ./ sigma;
    norm_adapted_val_features = (tuning_val_features - mu) ./ sigma;
    norm_adapted_test_features = (tuning_test_features - mu) ./ sigma;
end

% -------------------------------------------------------------------------
% Transfer Learning
% -------------------------------------------------------------------------
% Take Features from Cross-Data and Fine-Tune the SVM Feature Space with 
% new Cross-Data Information

if do_transfer_learning

    if do_transfer_learning && ~do_domain_adaptation

        if ~hyper
            
            % TRANSFER LEARNING w/o Domain Adaptation & Hyperparamter Tuned Model
            
            fprintf('\n\n [] Performing Fine-Tuning (without Domain Adaptation) using Cross-Dataset features: \n');

            loaded_model = load('24_4sec_proc3wRef_HEATCHAIR_model.mat');
            mdl_workload = loaded_model.mdl_workload;

            % Fine-Tune the existing model using the new dataset
            tuned_mdl_workload = fitcsvm([mdl_workload.X; tuning_train_features], [mdl_workload.Y; tuning_train_labels],...
                'KernelFunction', mdl_workload.KernelParameters.Function, 'BoxConstraint', mdl_workload.BoxConstraints(1));

            % Save the Updated Model only for Transfer Learning
%             save('24_4sec_proc3wRef_HEATCHAIR_finetuned_model.mat', 'tuned_mdl_workload');
%             fprintf('Fine-tuning complete. Fine-tuned Model saved.\n');

%             tuned_mdl_workload_loader = load('23_4sec_rawproc_STEW_finetuned_model.mat');
%             tuned_mdl_workload = tuned_mdl_workload_loader.tuned_mdl_workload;

            % Evaluation: Test Performance on Training-Data
            fprintf('Evaluating Fine-Tuned Model on Training-Data...\n');
            eval_mdl_performance(tuned_mdl_workload, val_features, val_labels, [], 'Fine-Tuned Train Dataset');

            % Evaluation: Test Performance on Cross-Data
            fprintf('Evaluating Fine-Tuned Model on Cross-Data...\n');
            eval_mdl_performance(tuned_mdl_workload, tuning_val_features, tuning_val_labels, [], 'Fine-Tuned Cross Dataset');

            fprintf('Fine-Tuned w/o Domain-Adaptation Evaluation Complete.\n');

        elseif hyper

            % HYPERPARAMETER & TRANSFER LEARNING w/o Domain Adaptation

            fprintf('\n\n [] Performing Fine-Tuning (without Domain Adaptation) using Cross-Dataset Features and Hyperparameter Tuned Model: \n');

            loaded_model = load('hyper_24_4sec_proc3wRef_HEATCHAIR_model.mat');
            mdl_workload = loaded_model.mdl_workload;

            % Fine-Tune the existing model using the new dataset
            tuned_mdl_workload = fitcsvm([mdl_workload.X; tuning_train_features], [mdl_workload.Y; tuning_train_labels],...
                'KernelFunction', mdl_workload.KernelParameters.Function, 'BoxConstraint', mdl_workload.BoxConstraints(1));

            % Save the Updated Model only for Transfer Learning
            save('hyper_24_4sec_proc3wRef_HEATCHAIR_finetuned_model.mat', 'tuned_mdl_workload');
            fprintf('Fine-tuning complete. Fine-tuned Model saved.\n');

%             tuned_mdl_workload_loader = load('23_4sec_rawproc_STEW_finetuned_model.mat');
%             tuned_mdl_workload = tuned_mdl_workload_loader.tuned_mdl_workload;

            % Evaluation: Test Performance on Training-Data
            fprintf('Evaluating Fine-Tuned Model on Training-Data...\n');
            eval_mdl_performance(tuned_mdl_workload, val_features, val_labels, [],'Fine-Tuned Train Dataset');

            % Evaluation: Test Performance on Cross-Data
            fprintf('Evaluating Fine-Tuned Model on Cross-Data...\n');
            eval_mdl_performance(tuned_mdl_workload, tuning_val_features, tuning_val_labels, [], 'Fine-Tuned Cross Dataset');

            fprintf('(Hyperparameter Tuned Model) Fine-Tuned w/o Domain-Adaptation Evaluation Complete.\n');

        end


    elseif do_domain_adaptation && do_transfer_learning

        if ~hyper

            % TRANSFER LEARNING & DOMAIN ADAPTATION w/o Hyperparameter Tuned Model
            
            fprintf('\n\n [] Performing Fine-Tuning using Domain-Adapted Cross-Dataset features: \n');

            norm_mdl_workload_loader = load('24_4sec_proc3wRef_HEATCHAIR_norm_model.mat');
            norm_mdl_workload = norm_mdl_workload_loader.norm_mdl_workload;

            % Fine-Tune the existing model using the new dataset
            tuned_adapted_mdl_workload = fitcsvm([norm_mdl_workload.X; norm_adapted_train_features], [norm_mdl_workload.Y; tuning_train_labels],...
                'KernelFunction', norm_mdl_workload.KernelParameters.Function, 'BoxConstraint', norm_mdl_workload.BoxConstraints(1));

            % Save Updated Model for Transfer Learning using Domain Adaptation
            save('24_4sec_proc3wRef_HEATCHAIR_finetuned_adapted_norm_model.mat', 'tuned_adapted_mdl_workload');
            fprintf('Fine-tuning with Domain Adaptation complete. Normalized Fine-tuned + Domain adapted Model saved.\n');

%             tuned_adapted_mdl_workload_loader = load('23_4sec_rawproc_STEW_finetuned_adapted_model.mat');
%             tuned_adapted_mdl_workload = tuned_adapted_mdl_workload_loader.tuned_adapted_mdl_workload;

            % Evaluate the new model on the training data
            fprintf('Evaluating the Transfer-Learning+Domain-Adapted Model on Training Data...\n');
            eval_mdl_performance(tuned_adapted_mdl_workload, norm_val_features, val_labels, [], 'Fine-Tuned Domain Adapted Train Dataset');

            % Evaluate the new model on the cross data
            fprintf('Evaluating the Transfer-Learning+Domain-Adapted Model on Cross Data...\n');
            eval_mdl_performance(tuned_adapted_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], 'Fine-Tuned Domain Adapted Cross Dataset');

            fprintf('(Normalized) Fine-Tuned + Domain-Adaptation Evaluation Complete.\n');

        elseif hyper

            % HYPERPARAMETER & TRANSFER LEARNING & DOMAIN ADAPTATION 

            fprintf('\n\n [] Performing Fine-Tuning using Domain-Adapted Cross-Dataset features: \n');

            norm_mdl_workload_loader = load('hyper_24_4sec_proc3wRef_HEATCHAIR_norm_model.mat');
            norm_mdl_workload = norm_mdl_workload_loader.norm_mdl_workload;

            % Fine-Tune the existing model using the new dataset
            tuned_adapted_mdl_workload = fitcsvm([norm_mdl_workload.X; norm_adapted_train_features], [norm_mdl_workload.Y; tuning_train_labels],...
                'KernelFunction', norm_mdl_workload.KernelParameters.Function, 'BoxConstraint', norm_mdl_workload.BoxConstraints(1));

            % Save Updated Model for Transfer Learning using Domain Adaptation
            save('24_4sec_proc3wRef_HEATCHAIR_finetuned_adapted_norm_model.mat', 'tuned_adapted_mdl_workload');
            fprintf('Fine-tuning with Domain Adaptation complete. Normalized Fine-tuned + Domain adapted Model saved.\n');

%             tuned_adapted_mdl_workload_loader = load('23_4sec_rawproc_STEW_finetuned_adapted_model.mat');
%             tuned_adapted_mdl_workload = tuned_adapted_mdl_workload_loader.tuned_adapted_mdl_workload;

            % Evaluate the new model on the training data
            fprintf('Evaluating the Transfer-Learning+Domain-Adapted Model on Training Data...\n');
            eval_mdl_performance(tuned_adapted_mdl_workload, norm_val_features, val_labels, [], 'Fine-Tuned Domain Adapted Train Dataset')

            % Evaluate the new model on the cross data
            fprintf('Evaluating the Transfer-Learning+Domain-Adapted Model on Cross Data...\n');
            eval_mdl_performance(tuned_adapted_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], 'Fine-Tuned Domain Adapted Cross Dataset');

            fprintf('(Hyperparameter Tuned Normalized Model) Fine-Tuned + Domain-Adaptation Evaluation Complete.\n');

        end

    end

end




%%  SAVE

%% Offline Processing Pipeline for pBCI Classification

% Offline Data Loading, Segmenting, pBCI Preprocessing and Feature
% Extraction and Model Training and Evaluation

% cd('E:\SchuleJobAusbildung\HTW\MasterThesis\Code\Matlab');

%% Information about Data Structure

% Hyperparameter Tuned = hyper (includes best_C and best_kernel)
% Amount of Features: 'allCH'(all Channels); '27'; '23'
% Epoch Time: '2sec'; '4sec'; 
% Raw Data: 'raw'
% Processed Data: 'processed'
% Processed from Download: 'rawproc' / 'raw'
% Processed with Average Reference; 1 to 40 Hz Butterworth 2nd order: 'proc1'
% Processed with Average Reference; 1 to 40 Hz Butterworth 2nd order; ASR Filtered: 'proc1ASR'
% Processed with Average Reference; 2 to 40 Hz Butterworth 2nd order; ASR Filtered: 'proc12ASR'
% Processed with Average Reference; 2 to 25 Hz Butterworth 2nd order: 'proc2'
% Processed with Average Reference; 2 to 25 Hz Butterworth 2nd order; ASR Filtered: 'proc2ASR'
% Processed with DC Offset Removal; 2 to 20 Hz Butterworth 2nd order; Selfwritten Artifact Removal; Average Referenced: 'proc3wRef;
% Processed with DC Offset Removal; 2 to 20 Hz Butterworth 2nd order; Selfwritten Artifact Removal; NO Average Referenced: 'proc3noRef;
% Dataset Name: 'STEW'; 'HEATCHAIR'; 'MATB' 
% Data Object Definition: 'epochs'; 'labels'; 'train/val/test_features';
% 'model'; 'norm_model' (for normalized Models); 'finetuned' (for Transfer
% Learning Models); 'finetuned_adapted' (for Transfer Learning with Domain
% Adaptation Data)


%% Setup EEGLAB GUI


addpath(genpath('E:\SchuleJobAusbildung\HTW\MasterThesis\Code\EEGLAB'));
eeglab;

%% Load EEG Dataset and segment into Epochs with corresponding Labels

fprintf('\n\n [] Loading Segmented EEG and Variables: \n');

% Load the segemented datasets + labels using the Segmentation Scripts
eeg_data_loader = load('4sec_raw_HEATCHAIR_epochs_v2.mat');
eeg_data = eeg_data_loader.eeg_data;


% Initialize Variables
fs = 128;                                           % Sampling Frequency
epoch_length = 4*fs;                                % Length of 1 Epoch in Seconds
overlap = 0.5;                                      % Amount of Overlap between Epochs
step_size = round((1-overlap) * epoch_length);      % Amount of Single Step Size in Samples


%% Preprocess Epochs (Referencing, Filtering, ASR)
% Apply Preprocessing to all datasets before splitting for training
% using my "preprocess_epochs" function

fprintf('\n\n [] (Loading) Preprocessing the EEG and Labels:  \n');

processed_eeg = OFF_preprocess_epochs(eeg_data,fs,true);

% Load processed EEG segments and related labels
loaded_processed_data = load('4sec_proc3noRef_STEW_epochs.mat');
eeg_data_processed = loaded_processed_data.processed_eeg;

% Load corresponding labels
loaded_labels = load('4sec_STEW_labels_v2.mat');
labels = loaded_labels.labels;


% % -------------------------------------------------------------------------
% % Quick Check if Average Referencing is applied
% % -------------------------------------------------------------------------
for ep = 1:size(eeg_data_processed, 3)
    eeg_data_avgref(:,:,ep) = eeg_data_processed(:,:,ep) - mean(eeg_data_processed(:,:,ep), 1);
end
mean_across_channels = mean(eeg_data_avgref, 1);                       % [1 × Time × Epochs]
mean_across_channels_epochs = squeeze(mean(mean_across_channels, 3));  % [1 × Time]

plot(mean_across_channels_epochs);
xlabel('Time (samples)');
ylabel('Amplitude (μV)');
title('Mean across channels and epochs (after avg ref)');
grid on;

% Just compare one epoch
original_processed = eeg_data_processed(:,:,1);
avgref = eeg_data_avgref(:,:,1);

figure;
subplot(2,1,1);
plot(original_processed(1,:)); title('Original - Channel 1');

subplot(2,1,2);
plot(avgref(1,:)); title('Avg Referenced - Channel 1');


% % -------------------------------------------------------------------------
%% Different Approach to check if Average Referencing is applied:

% ALSO CHECKS DIFFERENCE BETWEEN BEFORE AND AFTER PREPROCESSING

% (RE) Average Reference the dataset:
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

% % -------------------------------------------------------------------------


%% Randomly Split Data into (70% Training, 15% Validation, 15% Test)

fprintf('\n\n [] Splitting EEG and Labels into 70/15/15 Train, Val, Test: \n');

% Make sure to evenly split high and low classes
% Labels: 0 = Low, 1 = High
idx_low = find(labels == 0);
idx_high = find(labels == 1);

% Shuffle within class
rng(42);
idx_low = idx_low(randperm(length(idx_low)));
idx_high = idx_high(randperm(length(idx_high)));

% Get total per class (take the smaller one to balance)
min_class_count = min(length(idx_low), length(idx_high));

% Trim both classes to same size
idx_low = idx_low(1:min_class_count);
idx_high = idx_high(1:min_class_count);

% Now split each class into 70/15/15
num_train = round(0.7 * min_class_count);
num_val = round(0.15 * min_class_count);
num_test = min_class_count - num_train - num_val;

% LOW class split
low_train = idx_low(1:num_train);
low_val   = idx_low(num_train+1:num_train+num_val);
low_test  = idx_low(num_train+num_val+1:end);

% HIGH class split
high_train = idx_high(1:num_train);
high_val   = idx_high(num_train+1:num_train+num_val);
high_test  = idx_high(num_train+num_val+1:end);

% Merge train/val/test sets
train_idx = [low_train; high_train];
val_idx   = [low_val; high_val];
test_idx  = [low_test; high_test];

% Shuffle each split to avoid order bias
train_idx = train_idx(randperm(length(train_idx)));
val_idx   = val_idx(randperm(length(val_idx)));
test_idx  = test_idx(randperm(length(test_idx)));

% Final data
train_epochs = eeg_data_processed(:,:,train_idx);
val_epochs   = eeg_data_processed(:,:,val_idx);
test_epochs  = eeg_data_processed(:,:,test_idx);

train_labels = labels(train_idx);
val_labels   = labels(val_idx);
test_labels  = labels(test_idx);

downsampled_labels = [train_labels; val_labels; test_labels];

% Save labels (optional)
save('4sec_STEW_train_labels_v2.mat', "train_labels");
save('4sec_STEW_val_labels_v2.mat', "val_labels");
save('4sec_STEW_test_labels_v2.mat', "test_labels");

save('4sec_STEW_sampled_labels.mat', "downsampled_labels");


%% Feature Extraction

fprintf('\n\n [] Extracting Features: \n');

% Get CSP filters
num_csp_filters = 4;    % 2 from each class (should be sufficient and fast enough for real-time approach)
[W_csp, ~] = train_csp(train_epochs, train_labels, num_csp_filters);
save('csp_4sec_proc3noRef_STEW_features.mat', 'csp_features_train');

% Extract CSP Features
csp_train_features = extract_csp_features(train_epochs, W_csp);
csp_val_features   = extract_csp_features(val_epochs, W_csp);
csp_test_features  = extract_csp_features(test_epochs, W_csp);

% Save CSP features 
save('csp_4sec_proc3noRef_STEW_train_features.mat', 'csp_train_features');
save('csp_4sec_proc3noRef_STEW_val_features.mat', 'csp_val_features');
save('csp_4sec_proc3noRef_STEW_test_features.mat', 'csp_test_features');


% Extract Train Features
train_features = OFF_extract_features(train_epochs, epoch_length, fs, 24);
save('24_4sec_proc3noRef_STEW_train_features.mat', 'train_features');
 
% Extract Validation Features
val_features = OFF_extract_features(val_epochs, epoch_length, fs, 24);
save('24_4sec_proc3noRef_STEW_val_features.mat', 'val_features');

% Extract Test Features
test_features = OFF_extract_features(test_epochs, epoch_length, fs, 24);
save('24_4sec_proc3noRef_STEW_test_features.mat', 'test_features');


% Get CSP filters + existing features:
combined_features_train = [train_features, csp_train_features];
combined_features_val = [val_features, csp_val_features];
combined_features_test = [test_features, csp_test_features];
save('24wCsp_4sec_proc3noRef_STEW_train_features.mat', 'combined_features_train');
save('24wCsp_4sec_proc3noRef_STEW_val_features.mat', 'combined_features_val');
save('24wCsp_4sec_proc3noRef_STEW_test_features.mat', 'combined_features_test');


fprintf('Total Feature Extraction complete! Features saved.\n');


%% Classification

fprintf('\n\n [] Classification (Loading Train/Val/Test Features): \n');

% Load features
[features, ~, train_features, val_features, test_features] = get_data('STEW');


% Loading labels
fprintf('Loading labels...\n');
[~, labels, ~, ~, ~, train_labels, val_labels, test_labels] = get_data('STEW');


% Check for epoch-labels match
if size(train_features,1) ~= size(train_labels)
    error('Mismatch between number of epochs and labels. Check dataset alignment.');
end
if size(val_features,1) ~= size(val_labels)
    error('Mismatch between number of epochs and labels. Check dataset alignment.');
end
if size(test_features,1) ~= size(test_labels)
    error('Mismatch between number of epochs and labels. Check dataset alignment.');
end

% -------------------------------------------------------------------------
% Train Classification Model
% -------------------------------------------------------------------------

fprintf('\n\n [] Classification (Loading/ Training Model): \n');

mdl_workload = fitcsvm(train_features, train_labels, 'KernelFunction','linear');

fprintf('Training complete! Saving model...\n');
save('test_25wCsp_STEW_model.mat', 'mdl_workload');
fprintf('Model saved successfully.\n');

% Load saved model
loaded_model = load('24_4sec_proc3noRef_STEW_model.mat');
mdl_workload = loaded_model.mdl_workload;

% -------------------------------------------------------------------------
% Validate the model 
% -------------------------------------------------------------------------
fprintf('Validating the model...\n');
eval_mdl_performance(mdl_workload, val_features, val_labels, [], 'Validating Base Model')


% -------------------------------------------------------------------------
% Hyperparameter Tuning for SVM
% -------------------------------------------------------------------------
fprintf('\n\n [] Hyperparameter Tuning Model with Val Data: \n');

% Define Hyperparameter Search Space

% Define Values to try for each Hyperparameter
C_values = max(logspace(-3, 1, 5), eps);    % Regularization strength (0.001 to 10) - more conservative range to avoid complexity and overfitting
kernels = {'linear', 'polynomial'};         % Different SVM Kernels (not using 'rbf' as it is more prone to overfitting)
%gamma_values = logspace(-3, 3, 7);         % Only used for RBF Kernel

% Number of folds for cross-validation
numFolds = 5; % 5-fold cross validation

% Track best model
best_C = NaN;
best_kernel = '';
%best_gamma = 0;
best_accuracy = -Inf;

fprintf('Starting Hyperparameter Tuning using %d-fold cross-validation...\n', numFolds);

for k = 1:length(kernels)
    for c = 1:length(C_values)

        % Train SVM with these parameters
        mdl = fitcsvm(train_features, train_labels, 'KernelFunction', kernels{k}, 'BoxConstraint', C_values(c), 'KFold', numFolds);

        % Cross-validation accuracy
        cv_predictions = kfoldPredict(mdl);
        cv_accuracy = mean(cv_predictions == train_labels) * 100;

        % Update best parameters if accuracy improves
        if cv_accuracy > best_accuracy
            best_C = C_values(c);
            best_kernel = kernels{k};
            best_accuracy = cv_accuracy;
        end
    end
end

% -------------------------------------------------------------------------
% Train Final Model using best Hyperparameters
% -------------------------------------------------------------------------

% Ensure valid hyperparameters were found
if isempty(best_kernel) || isnan(best_C)
    error('Hyperparameter tuning failed: No valid model was found.');
end

fprintf('Best Model: Kernel = %s | C = %.5f | Accuracy = %.2f%%\n', best_kernel, best_C, best_accuracy);

% Train the new model with the best kernel and best C
mdl_workload = fitcsvm([train_features; val_features], [train_labels; val_labels], 'KernelFunction', best_kernel, 'BoxConstraint', best_C);

fprintf('Final model trained successfully on full training + validation set. \n');
save('hyper_24_4sec_proc3noRef_MATB_easy_diff_model.mat', 'mdl_workload', 'best_C', 'best_kernel');
fprintf('Saved Final Model.\n');

% Load saved model
loaded_model = load('hyper_24_4sec_proc3noRef_STEW_model.mat');
mdl_workload = loaded_model.mdl_workload;

% -------------------------------------------------------------------------
% Evaluate the model on completely unseen Test Set
% -------------------------------------------------------------------------
fprintf('Evaluating final model on the unseen Test set...\n');
eval_mdl_performance(mdl_workload, test_features, test_labels, [], 'Testing Hyperparameter Tuned Model')

%% Normalized Model 

% -------------------------------------------------------------------------
% Train a Normalized Model for Domain Adaptation
% -------------------------------------------------------------------------

% Normalize training dataset using training data statistics
mu = mean(train_features);
sigma = std(train_features);
sigma(sigma == 0) = 1;          % Avoid division by zero (if any feature has zero variance)

% Normalize the Training Data
norm_train_features = (train_features - mu) ./ sigma;
norm_val_features = (val_features - mu) ./ sigma;
norm_test_features = (test_features - mu) ./ sigma;

% Train the normalized model using norm_train_features
norm_mdl_workload = fitcsvm(norm_train_features, train_labels, 'KernelFunction', 'linear');
save('24_4sec_proc3noRef_STEW_norm_model.mat', 'norm_mdl_workload');
fprintf('Normalized Model saved successfully.\n');


% -------------------------------------------------------------------------
% Validation of Normalized Model
% -------------------------------------------------------------------------
fprintf('Validating the model...\n');
eval_mdl_performance(norm_mdl_workload, norm_val_features, val_labels, [], 'Normalized Model Validation')


% -------------------------------------------------------------------------
% Hyperparameter Tuning for Normalized SVM
% -------------------------------------------------------------------------

% Define Hyperparameter Search Space

% Define Values to try for each Hyperparameter
norm_C_values = max(logspace(-3, 1, 5), eps);    % Regularization strength (0.001 to 10) - more conservative range to avoid complexity and overfitting
norm_kernels = {'linear', 'polynomial'};         % Different SVM Kernels (not using 'rbf' as it is more prone to overfitting)
%gamma_values = logspace(-3, 3, 7);              % Only used for RBF Kernel

% Number of folds for cross-validation
numFolds = 5; % 5-fold cross validation

% Track best model
norm_best_C = NaN;
norm_best_kernel = '';
%best_gamma = 0;
norm_best_accuracy = -Inf;

fprintf('Starting Hyperparameter Tuning for Normalized Model using %d-fold cross-validation...\n', numFolds);

for k = 1:length(norm_kernels)
    for c = 1:length(norm_C_values)

        % Train SVM with these parameters
        mdl = fitcsvm(norm_train_features, train_labels, 'KernelFunction', norm_kernels{k},'BoxConstraint', norm_C_values(c) ,'KFold', numFolds);

        % Cross-validation accuracy
        norm_cv_predictions = kfoldPredict(mdl);
        norm_cv_accuracy = mean(norm_cv_predictions == train_labels) * 100;

        % Update best parameters if accuracy improves
        if norm_cv_accuracy > norm_best_accuracy
            norm_best_C = norm_C_values(c);
            norm_best_kernel = norm_kernels{k};
            norm_best_accuracy = norm_cv_accuracy;
        end
    end
end

% -------------------------------------------------------------------------
% Train Final Normalized Model using best Hyperparameters
% -------------------------------------------------------------------------

% Ensure valid hyperparameters were found
if isempty(norm_best_kernel) || isnan(norm_best_C)
    error('Hyperparameter tuning failed: No valid model was found.');
end

fprintf('Best Model: Kernel = %s | C = %.5f | Accuracy = %.2f%%\n', norm_best_kernel, norm_best_C, norm_best_accuracy);

% Train the Classifier with the best kernel and best C
norm_mdl_workload = fitcsvm([norm_train_features; norm_val_features], [train_labels; val_labels], 'KernelFunction', norm_best_kernel, 'BoxConstraint', norm_best_C);

fprintf('Final Normalized model trained successfully on full norm_training + norm_validation set. \n');
save('hyper_24_4sec_proc3noRef_STEW_norm_model.mat', 'norm_mdl_workload', 'norm_best_C', 'norm_best_kernel');
fprintf('Saved Final Normalized Hypertuned Model.\n');


% -------------------------------------------------------------------------
% Evaluate the Normalized model on (same dataset) Test Set
% -------------------------------------------------------------------------
fprintf('Evaluating final Normalized model on the unseen Test set...\n');
eval_mdl_performance(norm_mdl_workload, norm_test_features, test_labels, [], 'Hyperparameter Tuned Normalized Testing Evaluation')
fprintf('Hyperparameter Tuned Normalized Features Classification Evaluation Complete.\n');


%% Cross Dataset Testing - PRE CALIBRATION

fprintf('\n\n [] Starting Cross-Dataset Testing:\n')
fprintf('Loading pre-trained model for Cross-Dataset Testing...\n');

% Load trained model
loaded_model = load('24_4sec_proc3noRef_STEW_model.mat');
mdl_workload = loaded_model.mdl_workload;

% Load Hyperparameter Tuned model
hyper_loaded_model = load('hyper_24_4sec_proc3noRef_STEW_model.mat');
hyper_mdl_workload = hyper_loaded_model.mdl_workload;
best_C = hyper_loaded_model.best_C;
best_kernel = hyper_loaded_model.best_kernel;
fprintf('Model Loaded!')

% Load Current-Dateset Features
[~, ~, train_features] = get_data('HEATCHAIR');

% Load Cross-Dataset Features and Labels
[cross_features, cross_labels] = get_data('MATB_EASY_DIFF');

% Perform Prediction on the New Dataset
eval_mdl_performance(mdl_workload, cross_features, cross_labels, [], 'Cross Dataset Test Pre-Calibration')
fprintf('Cross-Dataset Evaluation Complete.\n');

% Perform Prediction on the New Dataset using Hyperparameter Tuned Model
eval_mdl_performance(hyper_mdl_workload, cross_features, cross_labels, [], 'Cross Dataset Test Pre-Calibration')
fprintf('Cross-Dataset Hyperparameter Tuned Evaluation Complete.\n');

% -------------------------------------------------------------------------
% Check Differences Between Dataset FEATURES - PRE CALIBRATION
% -------------------------------------------------------------------------
% PRE Transfer Learning and PRE Domain Adaptation
% Load Features from Both Datasets
fprintf('\n\n [] Checking Feature Distribution between Training and Cross Data: \n');

% Get Number of Features and Parameters
features_per_fig = 8;
num_features = size(train_features, 2);
num_figures = ceil(num_features/ features_per_fig);
feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', 'Theta Beta Ratio', 'Alpha Beta Ratio', 'Engagement Index',...
    'Theta Frontal','Theta Parietal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', 'Beta Temporal',...
    'Beta Parietal', 'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Avg Mobility', 'Avg Complexity',...
    'Avg Entropy', 'Theta Entropy', 'Alpha Entropy'};

if length(feature_names) ~= num_features
    error('Mismatch: Length of Feature Names unequal to Number of Features!')
end

% Plot Feature Distributions
for fig_idx = 1:num_figures
    figure(fig_idx);
    start_feature = (fig_idx - 1) * features_per_fig + 1;
    end_feature = min(fig_idx * features_per_fig, num_features);

    for i = start_feature:end_feature 
        subplot(2,4,i - start_feature + 1);
        histogram(train_features(:, i), 'Normalization', 'probability', 'FaceAlpha', 0.5);
        hold on;
        histogram(cross_features(:, i), 'Normalization', 'probability', 'FaceAlpha', 0.5);
        title(feature_names{i});
        xlabel('Feature Value');
        ylabel('Probability');
        legend('Original Dataset', 'Cross-Dataset');
    end

sgtitle(['Feature Distributions Comparison: Features ', num2str(start_feature), ' - ', num2str(end_feature)]);
hold off;
end

%% Preparing the Data for Cross-Data Calibration (Simulating the Calibration Phase of the Real-Time Experiment)

fprintf('\n\n [] Cross-Data Model Calibration and Evaluation: \n');

% -------------------------------------------------------------------------
% CROSS-DATA CALIBRATION 
% -------------------------------------------------------------------------




%% Performing Transfer Learning with/without Domain Adaptation using Standard and Hyperparameter Tuned Models

% % -------------------------------------------------------------------------
% % Settings for Calibration 
% % -------------------------------------------------------------------------
% % Data & Model Selection
% params.features = '24';                      % Amount of Features                    = '23', '24', '27'
% params.epoch = '4sec';                       % Epoch Length                          = '2sec', '4sec', '6sec'
% params.dataset = 'HEATCHAIR';                % Source/ Main/ Training Dataset        = 'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'
% params.calibrationset = 'MATB_easy_diff'; % Cross-Data Calibration Dataset        = 'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'

% % Calibration Options
% params.samples = 360;                        % Specify the amount of Cross-Data Samples for Calibration
% params.hyper = true;                        % Using Hyperparameter Tuned Models for Calibration
% params.only_domain_adaptation = false;        % Normalize Incoming Data using Source Statistics
% params.do_domain_adaptation = false;         % Normalize Incoming Data using the Source Statistics and pass it on for Transfer Learning
% params.do_transfer_learning = true;         % Add the Incoming Data to the Source Data and Retrain Classifier

% -------------------------------------------------------------------------
% Run Single Calibration 
% -------------------------------------------------------------------------
% run_calibration(params);

%%
% -------------------------------------------------------------------------
% Master Loop for All Calibration Cases
% -------------------------------------------------------------------------

dataset_names = {'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'};
calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};
results = init_results_table(dataset_names, calib_types);
numeric_results = NaN(height(results), width(results)); % Parallel matrix for cross-acc

for src = 1:length(dataset_names)
    for tgt = 1:length(dataset_names)
        if src == tgt, continue; end % Skip self-calibration

        for hyper = [false true]
            for calib_type = calib_types

                % --- Setup Calibration Params ---
                params.features = '24';
                params.epoch = '4sec';
                params.dataset = dataset_names{src};
                params.calibrationset = dataset_names{tgt};
                params.samples = 720;
                params.hyper = hyper;

                % Set calibration flags
                ct = calib_type{1};
                params.only_domain_adaptation = strcmp(ct, 'adapted');
                params.do_transfer_learning = strcmp(ct, 'finetuned') || strcmp(ct, 'finetuned_adapted');
                params.do_domain_adaptation = strcmp(ct, 'finetuned_adapted');

                fprintf('\n>> %s → %s | %s | Hyper: %s\n', ...
                    params.dataset, params.calibrationset, ct, string(hyper));

                % Run the calibration
                try
                    [acc1, acc2, calib_info, params] = run_calibration(params);
                catch ME
                    warning('[FAILED] %s → %s (%s) | Hyper: %s\n%s', ...
                        params.dataset, params.calibrationset, ct, string(hyper), ME.message);
                    acc1 = NaN; acc2 = NaN;
                    calib_info = struct('samples', params.samples, 'source_total', NaN, 'ratio', NaN);
                end

                % Update Results Table
                results = write_to_results_table(results, params, acc1, acc2, calib_info);
                row_idx = find(strcmp(results.Properties.RowNames, ...
                    sprintf('%s (Hyper: %s)', params.dataset, upper(string(params.hyper)))));
                col_idx = find(strcmp(results.Properties.VariableNames, ...
                    matlab.lang.makeValidName(sprintf('%s (%s)', params.calibrationset, params.calibration))));
                numeric_results(row_idx, col_idx) = acc2;  % Store cross accuracy
            end
        end
    end
end

% Use number of calibration samples in filename to compare future runs
sample_tag = sprintf('%dsamples', params.samples);  % e.g., 360

matfile   = ['all_calibration_results_' sample_tag '.mat'];
excelfile = ['all_calibration_results_' sample_tag '.xlsx'];

% Save results
save(matfile, 'results', 'numeric_results');

% Export the blockwise Excel
writetable(results, excelfile, 'WriteRowNames', true);

fprintf('[SAVED] %s and %s\n', matfile, excelfile);
disp('[DONE] All calibration combinations evaluated.');



%%

function [train_features, val_features, test_features] = extract_all_features(train_labels, val_labels, test_labels, train_epochs, val_epochs, test_epochs, opts)

    fs = 128;
    epoch_length = double(4*fs);
    feature_tag = '';
    train_features = []; val_features = []; test_features = [];

    if opts.use_features
        fprintf('\n[INFO] Extracting handcrafted features...\n');
        train_24 = OFF_extract_features(train_epochs, epoch_length, fs, opts.num_features);
        val_24   = OFF_extract_features(val_epochs, epoch_length, fs, opts.num_features);
        test_24  = OFF_extract_features(test_epochs, epoch_length, fs, opts.num_features);
        feature_tag = '24';
    end

    if opts.use_csp
        fprintf('\n[INFO] Extracting CSP features...\n');
        [W_csp, ~] = train_csp(train_epochs, train_labels, opts.num_csp_filters);
        csp_train = extract_csp_features(train_epochs, W_csp);
        csp_val   = extract_csp_features(val_epochs, W_csp);
        csp_test  = extract_csp_features(test_epochs, W_csp);

        if isempty(feature_tag)
            feature_tag = sprintf('csp_%d', opts.num_csp_filters);
            train_features = csp_train;
            val_features   = csp_val;
            test_features  = csp_test;
        else
            fprintf('\n[INFO] Combining handcrafted and CSP features...\n');
            feature_tag = sprintf('%swCsp', feature_tag);
            train_features = [train_24, csp_train];
            val_features   = [val_24, csp_val];
            test_features  = [test_24, csp_test];
        end
    else
        if opts.use_features
            train_features = train_24;
            val_features   = val_24;
            test_features  = test_24;
        end
    end

    % Save files
    base = sprintf('%s_%s_%s', feature_tag, opts.epochlength, opts.proc);
    save([base '_' opts.dataset '_train_features.mat'], 'train_features');
    save([base '_' opts.dataset '_val_features.mat'],   'val_features');
    save([base '_' opts.dataset '_test_features.mat'],  'test_features');

    % Check for epoch-labels match
if size(train_features,1) ~= size(train_labels)
    error('Mismatch between number of train epochs and train labels. Check dataset alignment.');
end
if size(val_features,1) ~= size(val_labels)
    error('Mismatch between number of val epochs and val labels. Check dataset alignment.');
end
if size(test_features,1) ~= size(test_labels)
    error('Mismatch between number of test epochs and test labels. Check dataset alignment.');
end

end

%%

function [features, labels, train_features, test_features, val_features, train_labels, val_labels, test_labels] = get_data(dataset_name)

switch dataset_name
    case 'MATB_easy_meddiff'
        % Load Cross-Dataset Features MATB EASY MEDDIFF
        loaded_features_train = load('24_4sec_proc3noRef_MATB_easy_meddiff_train_features.mat');
        loaded_features_val = load('24_4sec_proc3noRef_MATB_easy_meddiff_val_features.mat');
        loaded_features_test = load('24_4sec_proc3noRef_MATB_easy_meddiff_test_features.mat');
        train_features = loaded_features_train.train_features;
        val_features = loaded_features_val.val_features;
        test_features = loaded_features_test.test_features;
        features = [loaded_features_train.train_features; loaded_features_val.val_features; loaded_features_test.test_features];
        % Load Cross-Dataset Labels
        loaded_labels = load('4sec_MATB_easy_meddiff_sampled_labels.mat');
        train_labels_load = load('4sec_MATB_easy_meddiff_train_labels.mat');
        val_labels_load = load('4sec_MATB_easy_meddiff_val_labels.mat');
        test_labels_load = load('4sec_MATB_easy_meddiff_test_labels.mat');
        train_labels = train_labels_load.train_labels;
        val_labels = val_labels_load.val_labels;
        test_labels = test_labels_load.test_labels;
        labels = loaded_labels.downsampled_labels;

    case 'MATB_easy_diff'
        % Load Cross-Dataset Features MATB EASY DIFF
        loaded_features_train = load('24_4sec_proc3noRef_MATB_easy_diff_train_features.mat');
        loaded_features_val = load('24_4sec_proc3noRef_MATB_easy_diff_val_features.mat');
        loaded_features_test = load('24_4sec_proc3noRef_MATB_easy_diff_test_features.mat');
        train_features = loaded_features_train.train_features;
        val_features = loaded_features_val.val_features;
        test_features = loaded_features_test.test_features;
        features = [loaded_features_train.train_features; loaded_features_val.val_features; loaded_features_test.test_features];
        % Load Cross-Dataset Labels
        loaded_labels = load('4sec_MATB_easy_diff_sampled_labels.mat');
        train_labels_load = load('4sec_MATB_easy_diff_train_labels.mat');
        val_labels_load = load('4sec_MATB_easy_diff_val_labels.mat');
        test_labels_load = load('4sec_MATB_easy_diff_test_labels.mat');
        train_labels = train_labels_load.train_labels;
        val_labels = val_labels_load.val_labels;
        test_labels = test_labels_load.test_labels;
        labels = loaded_labels.downsampled_labels;

    case 'STEW'
        % Load Cross-Dataset Features STEW
        loaded_features_train = load('24_4sec_proc3noRef_STEW_train_features.mat');
        loaded_features_val = load('24_4sec_proc3noRef_STEW_val_features.mat');
        loaded_features_test = load('24_4sec_proc3noRef_STEW_test_features.mat');
        train_features = loaded_features_train.train_features;
        val_features = loaded_features_val.val_features;
        test_features = loaded_features_test.test_features;
        features = [loaded_features_train.train_features; loaded_features_val.val_features; loaded_features_test.test_features];
        % Load Cross-Dataset Labels
        loaded_labels = load('4sec_STEW_sampled_labels_v2.mat');
        labels = loaded_labels.downsampled_labels;
        train_labels_load = load('4sec_STEW_train_labels_v2.mat');
        val_labels_load = load('4sec_STEW_val_labels_v2.mat');
        test_labels_load = load('4sec_STEW_test_labels_v2.mat');
        train_labels = train_labels_load.train_labels;
        val_labels = val_labels_load.val_labels;
        test_labels = test_labels_load.test_labels;


    case 'HEATCHAIR'
        % Load Cross-Dataset Features HEATCHAIR
        loaded_features_train = load('24_4sec_proc3wRef_HEATCHAIR_train_features.mat');
        loaded_features_val = load('24_4sec_proc3wRef_HEATCHAIR_val_features.mat');
        loaded_features_test = load('24_4sec_proc3wRef_HEATCHAIR_test_features.mat');
        train_features = loaded_features_train.train_features;
        val_features = loaded_features_val.val_features;
        test_features = loaded_features_test.test_features;
        features = [loaded_features_train.train_features; loaded_features_val.val_features; loaded_features_test.test_features];
        % Load Cross-Dataset Labels
        loaded_labels = load('4sec_HEATCHAIR_sampled_labels_v2.mat');
        labels = loaded_labels.downsampled_4sec_HEATCHAIR_labels_v2;
        train_labels_load = load('4sec_HEATCHAIR_train_labels_v2.mat');
        val_labels_load = load('4sec_HEATCHAIR_val_labels_v2.mat');
        test_labels_load = load('4sec_HEATCHAIR_test_labels_v2.mat');
        train_labels = train_labels_load.train_labels;
        val_labels = val_labels_load.val_labels;
        test_labels = test_labels_load.test_labels;

end

% Check for matching dimensions
if size(features,1) ~= size(labels)
    error('Mismatch between Epochs and Labels Number!');
end

fprintf('[%s] dataset loaded successfully.\n', dataset_name);

end


%% working statistical results evaluation

function compute_accuracy_stats(output_name, opts)
    % Load classification results
    T = readtable(output_name);

    % Prepare arrays
    N = height(T);
    config_list = {};
    model_list = {};
    context_list = {};
    accuracy_list = [];

    for i = 1:N
        source = T.SOURCE{i};
        target = T.TARGET{i};
        acc = T.ACCURACY(i);
        if isnan(acc), continue; end

        % Detect context
        is_within = contains(target, 'Within', 'IgnoreCase', true);
        context = 'Within';
        if ~is_within, context = 'Cross'; end

        % Extract config and model
        tokens = strsplit(source, ' ');
        cfg = extract_cfg(tokens);
        mdl = extract_model(tokens);

        % Append
        config_list{end+1} = cfg;
        model_list{end+1} = mdl;
        context_list{end+1} = context;
        accuracy_list(end+1) = acc;
    end

    % Create summary table
    summaryT = table(config_list', model_list', context_list', accuracy_list', ...
        'VariableNames', {'Config', 'Model', 'Context', 'Accuracy'});

    % Compute grouped stats
    grouped = groupsummary(summaryT, {'Config','Model','Context'}, ...
        {'mean', 'std', 'median'}, 'Accuracy');
    grouped.Properties.VariableNames{'mean_Accuracy'} = 'MeanAccuracy';
    grouped.Properties.VariableNames{'std_Accuracy'} = 'StdAccuracy';
    grouped.Properties.VariableNames{'median_Accuracy'} = 'MedianAccuracy';

    % Sort model order
    model_order = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};
    grouped.Model = categorical(grouped.Model, model_order, 'Ordinal', true);
    grouped = sortrows(grouped, {'Config', 'Model', 'Context'});

    % Write results to Excel
    output_file = sprintf('%d_accuracy_stats.xlsx', opts.total_samples);
    writetable(grouped, output_file);
    fprintf('[INFO] Accuracy stats written to: %s\n', output_file);

    % ----- Boxplot -----
    figure('Name', 'Accuracy Boxplots', 'NumberTitle', 'off');
    subplot(1,2,1)
    boxplot(summaryT.Accuracy, summaryT.Config)
    title('Boxplot by Config')
    ylabel('Accuracy (%)')
    grid on

    subplot(1,2,2)
    boxplot(summaryT.Accuracy, summaryT.Model)
    title('Boxplot by Model Type')
    ylabel('Accuracy (%)')
    grid on

    % ----- 2-Way ANOVA -----
    % Only on "Cross" or all data (you can filter if needed)
    % Convert config and model to numeric grouping
    cfgs = unique(summaryT.Config);
    mdls = unique(summaryT.Model);

    cfg_map = containers.Map(cfgs, 1:length(cfgs));
    mdl_map = containers.Map(mdls, 1:length(mdls));

    cfg_idx = zeros(height(summaryT),1);
    mdl_idx = zeros(height(summaryT),1);
    for i = 1:height(summaryT)
        cfg_idx(i) = cfg_map(summaryT.Config{i});
        mdl_idx(i) = mdl_map(summaryT.Model{i});
    end

    % Run 2-way ANOVA without interaction plots
    fprintf('\n[INFO] Running 2-way ANOVA on Config × Model...\n');
    [p, tbl, stats] = anovan(summaryT.Accuracy, {cfg_idx, mdl_idx}, ...
        'model', 2, ...
        'varnames', {'Config', 'Model'}, ...
        'display', 'off');

    % Print p-values nicely
    fprintf('[RESULTS] 2-way ANOVA p-values:\n');
    fprintf('- Config:       p = %.4f\n', p(1));
    fprintf('- Model:        p = %.4f\n', p(2));
    fprintf('- Interaction:  p = %.4f\n', p(3));

    % Optional post-hoc comparison
    % multcompare(stats, 'Dimension', 1); % for Config
    % multcompare(stats, 'Dimension', 2); % for Model
end

function cfg = extract_cfg(tokens)
    valid_cfgs = {'24', 'csp', '24wCsp'};
    cfg = 'Unknown';
    for i = 1:length(tokens)
        if any(strcmpi(tokens{i}, valid_cfgs))
            cfg = lower(tokens{i});  % preserve lowercase for 'csp'
            return;
        end
    end
end

function mdl = extract_model(tokens)
    valid_mdls = {'STANDARD', 'HYPER', 'NORM', 'HYPER NORM'};
    mdl = 'Unknown';
    for i = 1:length(tokens)
        rest = strjoin(tokens(i:end), ' ');
        if any(strcmp(rest, valid_mdls))
            mdl = rest;
            return;
        elseif any(strcmp(tokens{i}, valid_mdls))
            mdl = tokens{i};
            return;
        end
    end
end





%% Automatic Transfer Learning with/without Domain Adaptation using Standard, Normalized and respective Hyperparameter Tuned Models

% -------------------------------------------------------------------------
% Master Loop for All Calibration Cases
% -------------------------------------------------------------------------
fprintf('\n\n [STAGE 8] Running Cross-Data Model Calibration and Evaluation... \n');

dataset_names = {'STEW', 'MATB_easy_meddiff', 'MATB_easy_diff', 'HEATCHAIR'};
calib_types = {'adapted', 'finetuned', 'finetuned_adapted'};
results = init_results_table(dataset_names, calib_types);
numeric_results = NaN(height(results), width(results)); % Parallel matrix for cross-acc

for src = 1:length(dataset_names)
    for tgt = 1:length(dataset_names)
        if src == tgt, continue; end % Skip self-calibration

        for hyper = [false true]
            for calib_type = calib_types

                % --- Setup Calibration Params ---
                params.features = '24';
                params.epoch = '4sec';
                params.dataset = dataset_names{src};
                params.calibrationset = dataset_names{tgt};
                params.samples = 720;
                params.hyper = hyper;

                % Set calibration flags
                ct = calib_type{1};
                params.only_domain_adaptation = strcmp(ct, 'adapted');
                params.do_transfer_learning = strcmp(ct, 'finetuned') || strcmp(ct, 'finetuned_adapted');
                params.do_domain_adaptation = strcmp(ct, 'finetuned_adapted');

                fprintf('\n>> %s → %s | %s | Hyper: %s\n', ...
                    params.dataset, params.calibrationset, ct, string(hyper));

                % Run the calibration
                try
                    [acc1, acc2, calib_info, params] = run_calibration(params);
                catch ME
                    warning('[FAILED] %s → %s (%s) | Hyper: %s\n%s', ...
                        params.dataset, params.calibrationset, ct, string(hyper), ME.message);
                    acc1 = NaN; acc2 = NaN;
                    calib_info = struct('samples', params.samples, 'source_total', NaN, 'ratio', NaN);
                end

                % Update Results Table
                results = write_to_results_table(results, params, acc1, acc2, calib_info);
                row_idx = find(strcmp(results.Properties.RowNames, ...
                    sprintf('%s (Hyper: %s)', params.dataset, upper(string(params.hyper)))));
                col_idx = find(strcmp(results.Properties.VariableNames, ...
                    matlab.lang.makeValidName(sprintf('%s (%s)', params.calibrationset, params.calibration))));
                numeric_results(row_idx, col_idx) = acc2;  % Store cross accuracy
            end
        end
    end
end

% Use number of calibration samples in filename to compare future runs
sample_tag = sprintf('%dsamples', params.samples);  % e.g., 360

matfile   = ['all_calibration_results_' sample_tag '.mat'];
excelfile = ['all_calibration_results_' sample_tag '.xlsx'];

% Save results
save(matfile, 'results', 'numeric_results');

% Export the blockwise Excel
writetable(results, excelfile, 'WriteRowNames', true);

fprintf('[SAVED] %s and %s\n', matfile, excelfile);

fprintf('\n============================================\n')
disp('[DONE] All Calibration Combinations Evaluated');
disp('[DONE] Calibration Phase Completed.');





%% 

function [acc1, acc2, calib_info, params] = run_calibration(params)

% -------------------------------------------------------------------------
% CROSS-DATA CALIBRATION
% -------------------------------------------------------------------------
close all;
fprintf('\n\n[INFO] Starting Cross-Data Model Calibration and Evaluation... \n');

% -------------------------------------------------------------------------
% Auto-set `params.proc` based on dataset
% -------------------------------------------------------------------------
% params.proc = 'proc3wRef';                   % Processing Type                  = 'proc3noRef','proc3wRef'
if strcmp(params.dataset, 'HEATCHAIR')
    params.proc = 'proc3wRef';
else
    params.proc = 'proc3noRef';
end

% -------------------------------------------------------------------------
% Inference of Calibration Type and Model Type
% -------------------------------------------------------------------------
%params.calibration = 'adapted';              % Calibration Type                  = 'adapted', 'finetuned', 'finetuned_adapted'
%params.modeltype = 'norm_model';             % Standard or Normalized Model Type = 'model', 'norm_model'

if params.only_domain_adaptation
    params.calibration = 'adapted';
    params.modeltype = 'norm_model';
elseif params.do_domain_adaptation && params.do_transfer_learning
    params.calibration = 'finetuned_adapted';
    params.modeltype = 'norm_model';
elseif params.do_transfer_learning && ~params.do_domain_adaptation
    params.calibration = 'finetuned';
    params.modeltype = 'model';
end

% -------------------------------------------------------------------------
% Calibration Data Selection
% -------------------------------------------------------------------------
% Selecting data from the train and cross-dataset to test for calibration

% Get Train Data
[~, ~, train_features, val_features, test_features, ~, val_labels, test_labels] = get_data(params.dataset);

% Get Calibration Data
[cross_features, cross_labels] = get_data(params.calibrationset);

% Select Calibration Subset (as per RT Pipeline Calibration Phase)
% Specify Calibration Samples Needed:
samples_per_class = params.samples / 2; % Half of Calibration Samples (for each class one half)
train_ratio = 0.7;
total_required_samples = round(params.samples / train_ratio);
samples_per_class_total = round(total_required_samples / 2);
val_test_samples_per_class = round((samples_per_class_total - samples_per_class)/2);

% Take small Sample Size of the Cross-Data to use for fine-tuning
low_idx = find(cross_labels == 0);
high_idx = find(cross_labels == 1);
rng(42);
low_idx = low_idx(randperm(length(low_idx)));
high_idx = high_idx(randperm(length(high_idx)));

% Split 70% / 15% / 15%
train_idx = [low_idx(1:samples_per_class); high_idx(1:samples_per_class)];
val_idx = [low_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class); high_idx(samples_per_class+1:samples_per_class+val_test_samples_per_class)];
test_idx = [low_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total); high_idx(samples_per_class+val_test_samples_per_class+1:samples_per_class_total)];
train_idx = train_idx(randperm(length(train_idx)));
val_idx = val_idx(randperm(length(val_idx)));
test_idx = test_idx(randperm(length(test_idx)));

% Training Data
tuning_train_features = cross_features(train_idx,:);
tuning_train_labels = cross_labels(train_idx);

% Validation and Testing Data (remaining unseen data)
tuning_val_features = cross_features(val_idx,:);
tuning_val_labels = cross_labels(val_idx);

% For now not using the test data - will be used if further 
% Hyperparameter Tuning included
tuning_test_features = cross_features(test_idx,:);
tuning_test_labels = cross_labels(test_idx);

% Display structure
fprintf('\n=== Final Stratified Split (Fully Balanced) ===\n');
fprintf('Train: %d samples |  Low: %d | High: %d\n', ...
    length(tuning_train_labels), sum(tuning_train_labels==0), sum(tuning_train_labels==1));
fprintf('Val:   %d samples |  Low: %d | High: %d\n', ...
    length(tuning_val_labels), sum(tuning_val_labels==0), sum(tuning_val_labels==1));
fprintf('Test:  %d samples |  Low: %d | High: %d\n', ...
    length(tuning_test_labels), sum(tuning_test_labels==0), sum(tuning_test_labels==1));


% -------------------------------------------------------------------------
% Build Model Filename
% -------------------------------------------------------------------------
prefix = '';
if params.hyper
    prefix = 'hyper_';
end
base_model_tag = [params.features '_' params.epoch '_' params.proc];
model_filename = [prefix base_model_tag '_' params.dataset '_' params.modeltype '.mat'];

fprintf('\n[INFO] Loading base model: %s\n', model_filename);
model_data = load(model_filename);

% Automatically detect the model variable (must contain "mdl" in its name)
model_vars = fieldnames(model_data);
mdl_var = model_vars(contains(model_vars, 'mdl'));

if isempty(mdl_var)
    error('[ERROR] No variable containing "mdl" found in %s', model_filename);
elseif length(mdl_var) > 1
    warning('[WARNING] Multiple variables with "mdl" found in %s. Using the first one: %s', ...
        model_filename, mdl_var{1});
end

mdl_workload = model_data.(mdl_var{1});

% -------------------------------------------------------------------------
% Normalize Features (if needed) - z-score
% -------------------------------------------------------------------------
if params.only_domain_adaptation || params.do_domain_adaptation
    fprintf('[INFO] Computing normalization statistics from training data...\n');
    mu = mean(train_features);
    sigma = std(train_features);
    sigma(sigma == 0) = 1;
    
    % Source Data Normalization
    norm_val_features = (val_features - mu) ./ sigma;
    norm_test_features = (test_features - mu) ./ sigma;

    % Cross-Data Normalization
    norm_adapted_train_features = (tuning_train_features - mu) ./ sigma;
    norm_adapted_val_features = (tuning_val_features - mu) ./ sigma;
    norm_adapted_test_features = (tuning_test_features - mu) ./ sigma;
end

% -------------------------------------------------------------------------
% Domain Adaptation Only (Evaluation only)
% -------------------------------------------------------------------------
if params.only_domain_adaptation
    fprintf('[INFO] Performing Domain Adaptation evaluation only...\n');

    if params.hyper
        fprintf('[INFO] Evaluating Hyperparameter Tuned Model after Domain Adaptation...\n');

        hyper_model_filename = ['hyper_' base_model_tag '_' params.dataset '_' params.modeltype '.mat'];
        hyper_model_data = load(hyper_model_filename);
        mdl_vars = fieldnames(hyper_model_data);
        hyper_mdl_var = mdl_vars{contains(mdl_vars, 'mdl', 'IgnoreCase', true)};
        hyper_mdl_workload = hyper_model_data.(hyper_mdl_var);

        eval_mdl_performance(hyper_mdl_workload, norm_adapted_val_features, tuning_val_labels, [], ...
            'DA: Hyperparameter Tuned Cross-Data Prediction');
        predictions = predict(hyper_mdl_workload, norm_adapted_val_features);
        acc2 = mean(predictions == tuning_val_labels);
        export_log(params, [], acc2);
    else
        eval_mdl_performance(mdl_workload, norm_adapted_val_features, tuning_val_labels, [], ...
            'DA: Cross-Data Prediction');
        predictions = predict(mdl_workload, norm_adapted_val_features);
        acc2 = mean(predictions == tuning_val_labels);
        export_log(params, [], acc2);
    end
end

% -------------------------------------------------------------------------
% Transfer Learning (Fine-Tuning)
% -------------------------------------------------------------------------
if params.do_transfer_learning
    fprintf('[INFO] Performing Transfer Learning...\n');

    % ---------------------- Consistency Check ----------------------------
    expected_calib = 'finetuned';
    if params.do_domain_adaptation
        expected_calib = 'finetuned_adapted';
    end
    if ~strcmp(expected_calib, params.calibration)
        warning('Calibration type in params.calibration ("%s") does not match detected configuration ("%s")', ...
            params.calibration, expected_calib);
    end

    if params.do_domain_adaptation
        fprintf('[INFO] Using Domain Adapted Cross-Data for Fine-Tuning...\n');

        if ~params.hyper
            finetune_features = norm_adapted_train_features;
            source_val_features = norm_val_features;
            source_val_tag = '[TL+DA] Fine-Tuned Domain Adapted TRAIN Dataset';
            cross_val_features = norm_adapted_val_features;
            cross_val_tag = '[TL+DA] Fine-Tuned Domain Adapted CROSS Dataset';
        else
            finetune_features = norm_adapted_train_features;
            source_val_features = norm_test_features;       % Select Test Features and Labels when using Hyperparameter Tuned
            source_val_tag = '[HYP+TL+DA] Hyperparameter Model Fine-Tuned Domain Adapted TRAIN Dataset';
            cross_val_features = norm_adapted_val_features;
            cross_val_tag = '[HYP+TL+DA] Hyperparameter Model Fine-Tuned Domain Adapted CROSS Dataset';
        end

        calib_type_tag = 'finetuned_adapted';
    else
        fprintf('[INFO] Using Raw Cross-Data for Fine-Tuning...\n');

        if ~params.hyper
            finetune_features = tuning_train_features;
            source_val_features = val_features;
            source_val_tag = '[TL] Fine-Tuned TRAIN Dataset';
            cross_val_features = tuning_val_features;
            cross_val_tag = '[TL] Fine-Tuned CROSS Dataset';
        else
            finetune_features = tuning_train_features;
            source_val_features = test_features;            % Select Test Features and Labels when using Hyperparameter Tuned
            source_val_tag = '[HYP+TL] Hyperparameter Model Fine-Tuned TRAIN Dataset';
            cross_val_features = tuning_val_features;
            cross_val_tag = '[HYP+TL] Hyperparameter Model Fine-Tuned CROSS Dataset';
        end

        calib_type_tag = 'finetuned';
    end

    % Fine-tune the existing model
    new_model = fitcsvm([mdl_workload.X; finetune_features], ...
        [mdl_workload.Y; tuning_train_labels], ...
        'KernelFunction', mdl_workload.KernelParameters.Function, ...
        'BoxConstraint', mdl_workload.BoxConstraints(1));

    % Save updated model
    final_model_name = [prefix base_model_tag '_' params.dataset '_' calib_type_tag '_' params.modeltype '.mat'];
    save(final_model_name, 'new_model');
    fprintf('[INFO] Saved new model: %s\n', final_model_name);

    if params.hyper
        source_val_labels = test_labels;
        source_val_features = test_features;
    else
        source_val_labels = val_labels;
        source_val_features = val_features;
    end

    % Evaluate
    eval_mdl_performance(new_model, source_val_features, source_val_labels, [], source_val_tag);
    eval_mdl_performance(new_model, cross_val_features, tuning_val_labels, [], cross_val_tag);

    predictions_source = predict(new_model, source_val_features);
    acc1 = mean(predictions_source == source_val_labels);

    predictions_cross = predict(new_model, cross_val_features);
    acc2 = mean(predictions_cross == tuning_val_labels);

    export_log(params, acc1, acc2);

    fprintf('[INFO] Domain Adaptation / Transfer Learning Pipeline completed.\n');

end


% -------------------------------------------------------------------------
% Return values (acc1, acc2, calib_info, params)
% -------------------------------------------------------------------------

% Ensure accuracy values exist
if ~exist('acc1', 'var'), acc1 = NaN; end
if ~exist('acc2', 'var'), acc2 = NaN; end

% Collect calibration info
calib_info.samples = params.samples;
calib_info.source_total = size(train_features, 1);
calib_info.ratio = calib_info.samples / calib_info.source_total;










% artifact removal oscar OFFLINE
% Precompute filter coefficients
% 4th order butterworth filter is too high, because we dont have enough samples to do forward and backward filtering
% Therefore we use 2nd order= smoother transitions, less aggressive filtering
% Low Cutoff 2 to prevent more slow drift artifacts in the signal
% High Cutoff 25 to prevent EMG artifacts

% Load Channel Location elp file once.
% persistent chanlocs
% if isempty(chanlocs)
%     loc_file = '14_ch_layout.locs';
%     if ~isfile(loc_file)
%         elp_file = fullfile(fileparts(which('eeglab.m')), 'plugins', 'dipfit', 'standard_BESA', 'Standard-10-5-cap385.elp');
%         locs = readlocs(elp_file);
%         electrode_labels = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
%         filtered_locs = locs(ismember({locs.labels}, electrode_labels));
%         writelocs(filtered_locs, loc_file);
%     end
%     chanlocs = readlocs(loc_file);
% end
% 
% % Setup EEG_templat to reuse EEGLAB structure in the loop
% EEG_template = eeg_emptyset();   % creates an empty EEGLAB dataset
% EEG_template.srate = fs;
% EEG_template.nbchan = size(raw_epochs,1);
% EEG_template.trials = 1;
% EEG_template.chanlocs = chanlocs;

    % Prepare variables for OSCAR processing with EEGLAB toolbox
    %EEG = EEG_template;
    %EEG.data = epoch_data_filtered;
    %EEG.pnts = size(epoch_data_filtered, 2);

    % Artifact Removal using OSCAR Live (EEGLAB Toolbox)
    % zThreshold = 7 (moderate threshold, reduce if stricter artifact rejection needed)
    % applyCAR = false (manually control referencing)
    %EEG = oscar(EEG, 'windowLength', 0.5, 'overlap', 0.25, 'zThreshold', 7,...
    %    'numSamples', fs, 'channelLocationFile', '', 'applyCAR', false, 'verbose', false);


% now RT
% Artifact Removal using OSCAR Live (EEGLAB Toolbox)
% zThreshold = 7 (moderate threshold, reduce if stricter artifact rejection needed)
% applyCAR = false (manually control referencing)
%EEG = oscar(EEG, 'windowLength', 0.5, 'overlap', 0.25, 'zThreshold', 7,...
%    'numSamples', fs, 'channelLocationFile', '', 'applyCAR', false, 'verbose', false);










