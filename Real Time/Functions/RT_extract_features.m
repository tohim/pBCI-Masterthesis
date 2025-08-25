function features = RT_extract_features(eeg_data_processed, epoch_length, fs, num_features)

% -------------------------------------------------------------------------
% Initializing
% -------------------------------------------------------------------------

% Initialize Parameters
nbchan = size(eeg_data_processed,1);
total_num_epochs = size(eeg_data_processed,3);

% Initialize Feature Matrix
features = zeros(total_num_epochs,num_features);                        

% Frequency Vector
f = linspace(0, fs/2, epoch_length/2);

% Frequency Bands
theta_band = [4, 8];
alpha_band = [8, 12];
beta_band = [13, 20];   % taking a Low Beta Range in an attempt to naturally avoid Muscle Artifacts (EMG)

% Compute Frequency Band Spectra
theta_idx = f >= theta_band(1) & f <= theta_band(2);
alpha_idx = f >= alpha_band(1) & f <= alpha_band(2);

% Electrode Numbers for each Brain Region
% For STEW recording
frontal_channels = [1, 2, 3, 12, 13, 14];   % AF3 = 1; F7 = 2; F3 = 3; F4 = 12; F8 = 13; AF4 = 14;
temporal_channels = [4, 5, 10, 11];         % FC5 = 4; T7 = 5; T8 = 10; FC6 = 11;
parietal_channels = [6, 9];                 % P7 = 6; P8 = 9;
posterior_channels = [7, 8];                % O1 = 7; O2 = 8;



% -------------------------------------------------------------------------
% Compute FFT for Power Spectral Density
% -------------------------------------------------------------------------
    epoch_fft = fft(eeg_data_processed, [], 2);             % FFT over time dimension
    fft_spectrum = abs(epoch_fft / epoch_length);           % Full sided (double) spectrum
    fft_spectrum = fft_spectrum(:,1:epoch_length/2);        % Taking half of the spectrum
    fft_spectrum(:, 2:end-1) = 2*fft_spectrum(:,2:end-1);   % Amplitude Correction
    psd = fft_spectrum .^2;                                 % Convert to Power Spectrum

    % Store brain region specific fft spectra
    frontal_psd = psd(frontal_channels,:);
    temporal_psd = psd(temporal_channels,:);
    parietal_psd = psd(parietal_channels,:);
    posterior_psd = psd(posterior_channels,:);


% -------------------------------------------------------------------------
% Saving mean Band Power over all channels
% -------------------------------------------------------------------------
    theta_power = median(arrayfun(@(ch) bandpower(psd(ch,:), f, theta_band, 'psd'), 1:nbchan));
    alpha_power = median(arrayfun(@(ch) bandpower(psd(ch,:), f, alpha_band, 'psd'), 1:nbchan));
    beta_power = median(arrayfun(@(ch) bandpower(psd(ch,:), f, beta_band, 'psd'), 1:nbchan));

    % Log scaling them to focus on relative changes instead of absolute values (reduces skewness)
    theta_power = log(1 + theta_power);
    alpha_power = log(1 + alpha_power);
    beta_power = log(1 + beta_power);
    

% -------------------------------------------------------------------------
    % Compute Region Specific Band Power
% -------------------------------------------------------------------------
    % (arrayfunction goes over all channels in the corresponding region specific fft)
    theta_frontal = median(arrayfun(@(ch) bandpower(frontal_psd(ch,:), f, theta_band, 'psd'), 1:size(frontal_psd,1)));
    theta_temporal = median(arrayfun(@(ch) bandpower(temporal_psd(ch,:), f, theta_band, 'psd'), 1:size(temporal_psd,1)));
    theta_parietal = median(arrayfun(@(ch) bandpower(parietal_psd(ch,:), f, theta_band, 'psd'), 1:size(parietal_psd,1)));
    theta_occipital = median(arrayfun(@(ch) bandpower(posterior_psd(ch,:), f, theta_band, 'psd'), 1:size(posterior_psd,1)));
    alpha_frontal = median(arrayfun(@(ch) bandpower(frontal_psd(ch,:), f, alpha_band, 'psd'), 1:size(frontal_psd,1)));
    alpha_temporal = median(arrayfun(@(ch) bandpower(temporal_psd(ch,:), f, alpha_band, 'psd'), 1:size(temporal_psd,1))); 
    alpha_parietal = median(arrayfun(@(ch) bandpower(parietal_psd(ch,:), f, alpha_band, 'psd'), 1:size(parietal_psd,1)));
    alpha_occipital = median(arrayfun(@(ch) bandpower(posterior_psd(ch,:), f, alpha_band, 'psd'), 1:size(posterior_psd,1)));
    beta_frontal = median(arrayfun(@(ch) bandpower(frontal_psd(ch,:), f, beta_band, 'psd'), 1:size(frontal_psd,1)));
    beta_temporal = median(arrayfun(@(ch) bandpower(temporal_psd(ch,:), f, beta_band, 'psd'), 1:size(temporal_psd,1)));
    beta_parietal = median(arrayfun(@(ch) bandpower(parietal_psd(ch,:), f, beta_band, 'psd'), 1:size(parietal_psd,1)));


% -------------------------------------------------------------------------
% Compute Engagement Index (Beta / (Alpha + Theta) for Frontal Power
% -------------------------------------------------------------------------
    % Higher EI = More task-focused and alert / Lower EI = fatigued              
    engagement_index = beta_frontal / (alpha_frontal + theta_frontal + eps); 

% -------------------------------------------------------------------------
% Compute Power Ratios for Enhanced Discrimination
% -------------------------------------------------------------------------
    % Increases under High MWL
    theta_alpha_ratio = theta_frontal ./ (alpha_parietal + alpha_occipital + eps);      % Cognitive Effort & Attention Load

    % Increases under Higher Mental Fatigue and Lower Attentional Control
    theta_beta_ratio = theta_frontal ./ (beta_frontal + eps);                           % Cognitive Effort & Alertness

    % Log scaling them to focus on relative changes instead of absolute values
    theta_alpha_ratio = log(1 + theta_alpha_ratio);
    theta_beta_ratio = log(1 + theta_beta_ratio);
    engagement_index = log(1 + engagement_index);

    theta_frontal = log(1 + theta_frontal);
    theta_temporal = log(1 + theta_temporal);
    theta_parietal = log(1 + theta_parietal);
    theta_occipital = log(1 + theta_occipital);
    alpha_frontal = log(1 + alpha_frontal);
    alpha_temporal = log(1 + alpha_temporal); 
    alpha_parietal = log(1 + alpha_parietal);
    alpha_occipital = log(1 + alpha_occipital);
    beta_frontal = log(1 + beta_frontal);
    beta_temporal = log(1 + beta_temporal);
    beta_parietal = log(1 + beta_parietal);


% -------------------------------------------------------------------------
% Compute Functional Connectivity 
% -------------------------------------------------------------------------
    % Initialize Coherence and PLV Matrices
    coherence_matrix = zeros(nbchan, nbchan);
    theta_coh_matrix = zeros(nbchan, nbchan);
    alpha_coh_matrix = zeros(nbchan, nbchan);
    %beta_coh_matrix = zeros(nbchan, nbchan);
    plv_matrix = zeros(nbchan, nbchan);

    % Coherence Parameters
    window_length = 256;
    coherence_overlap = round(window_length/4);  % Use only 25% overlap for more computational efficiency

    % Phase Locking Parameters
    as = hilbert(eeg_data_processed')';           % compute the analytic signal using Hilbert Transform
    phase_data = angle(as);                       % Computing instantaneous phase from as

    for j = 1:nbchan-1      % Reduce loops (only going up to nbchan-1)
        for k = j+1:nbchan  % Only compute for unique pairs of channels

            % Functional Connectivity with Coherence (a measure of how
            % synchronized 2 signals are in the fq domain, ranging from 0 to 1.
            % 0 -> no synchronization; 1 -> perfect synchronization)
            % Taking Coherence over full band filtered signal
            [Cxy, F] = mscohere(eeg_data_processed(j,:), eeg_data_processed(k,:), hamming(window_length), coherence_overlap, window_length, fs);

            % Computing the according Coherence Matrices
            coherence_matrix(j,k) = mean(Cxy);
            coherence_matrix(k,j) = coherence_matrix(j,k);  % Symmetric matrix

            theta_coh_matrix(j,k) = mean(Cxy(theta_idx));   % Theta Band
            theta_coh_matrix(k,j) = theta_coh_matrix(j,k);

            alpha_coh_matrix(j,k) = mean(Cxy(alpha_idx));
            alpha_coh_matrix(k,j) = alpha_coh_matrix(j,k);
            %beta_coh_matrix(j,k) = mean(beta_coh);
            %beta_coh_matrix(k,j) = beta_coh_matrix(j,k);

            % Compute Functional Connectivity with Phase Locking Values (PLV)
            phase_diff = phase_data(j,:) - phase_data(k,:);     % Compute pairwise phase difference
            plv_matrix(j,k) = abs(mean(exp(1i * phase_diff)));  % Compute PLV
            plv_matrix(k,j) = plv_matrix(j,k);

        end
    end

    avg_coherence = mean(coherence_matrix(:));      % captures synchronized oscillations across channels and all frequencies
    theta_coherence = mean(theta_coh_matrix(:));    % synchronized oscillations across all channels and theta band
    alpha_coherence = mean(alpha_coh_matrix(:));    % across all ch alpha band 
    

% -------------------------------------------------------------------------
% Hjorth Parameters to capture signal dynamics
% -------------------------------------------------------------------------
    diff1 = diff(eeg_data_processed,1,2);
    diff2 = diff(diff1,1,2);
    avg_mobility = mean(std(diff1,0,2) ./ std(eeg_data_processed,0,2));
    avg_complexity = mean(std(diff2,0,2) ./ std(diff1,0,2));


% -------------------------------------------------------------------------    
% Spectral Entropy (Total and for individual Frequency Bands)
% -------------------------------------------------------------------------
% Additional complexity measures
    normalized_spectrum = fft_spectrum ./ sum(fft_spectrum,2);
    spectrum_entropy = -sum(normalized_spectrum .* log(normalized_spectrum + eps), 2, 'omitnan');   % Shannon Entropy

% Compute band specific entropy
    % with log2 instead of log (makes entropy values more interpretable in "bits" (information theory perspective)
    theta_entropy = -sum((normalized_spectrum(:,theta_idx)) .* log2(normalized_spectrum(:,theta_idx) + eps), 2, 'omitnan');
    alpha_entropy = -sum((normalized_spectrum(:,alpha_idx)) .* log2(normalized_spectrum(:,alpha_idx) + eps), 2, 'omitnan');

    avg_entropy = mean(spectrum_entropy);
    theta_entropy = mean(theta_entropy);
    alpha_entropy = mean(alpha_entropy);


% -------------------------------------------------------------------------
% Store extracted features in Feature Vector
% -------------------------------------------------------------------------

% 25 Handcrafted Features
    features(:) = [theta_power, alpha_power, beta_power, theta_alpha_ratio, theta_beta_ratio, engagement_index, theta_frontal, ...
        theta_temporal, theta_parietal, theta_occipital, alpha_frontal, alpha_temporal, alpha_parietal, alpha_occipital, ...
        beta_frontal, beta_temporal, beta_parietal, avg_coherence, theta_coherence, alpha_coherence, avg_mobility, ...
        avg_complexity, avg_entropy, theta_entropy, alpha_entropy];

% Removin the 10 Worst Features from v1 -> NumFeatures = 14 (v2)
%     features(:) = [theta_power, alpha_power, beta_power, theta_alpha_ratio, theta_frontal, theta_parietal,...
%         alpha_frontal, alpha_parietal, alpha_occipital, beta_frontal, beta_temporal, beta_parietal, theta_entropy, alpha_entropy];
    

end

