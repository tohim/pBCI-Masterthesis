%% FEATURE INSPECTION

% Basic Feature Visualization
% Advanced Feature Visualization
% Statistical Tests on Feature Significance and Effect Size


%% Load Feature Names

% Feature Names
feature_names_allCh = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Frontal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal',...
                 'Alpha Posterior', 'Engagement Index', 'Coherence', 'Mobility', 'Complexity', 'Entropy', 'ch1_theta', 'ch2_theta',...
                 'ch3_theta', 'ch4_theta', 'ch5_theta', 'ch6_theta', 'ch7_theta', 'ch8_theta', 'ch9_theta', 'ch10_theta', 'ch11_theta',...
                 'ch12_theta', 'ch13_theta', 'ch14_theta', 'ch1_alpha', 'ch2_alpha', 'ch3_alpha', 'ch4_alpha', 'ch5_alpha', 'ch6_alpha',...
                 'ch7_alpha', 'ch8_alpha', 'ch9_alpha','ch10_alpha', 'ch11_alpha', 'ch12_alpha', 'ch13_alpha', 'ch14_alpha', 'ch1_beta',...
                 'ch2_beta', 'ch3_beta', 'ch4_beta', 'ch5_beta', 'ch6_beta', 'ch7_beta', 'ch8_beta', 'ch9_beta', 'ch10_beta',...
                 'ch11_beta', 'ch12_beta', 'ch13_beta', 'ch14_beta'};

feature_names_simple = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Frontal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal',...
                 'Alpha Posterior', 'Engagement Index', 'Coherence', 'Mobility', 'Complexity', 'Entropy'};

feature_names_27 = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', 'Theta Beta Ratio', 'Alpha Beta Ratio', 'Engagement Index',...
    'Theta Frontal','Theta Parietal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', 'Beta Temporal',...
    'Beta Parietal', 'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Beta Coherence', 'Avg PLV', 'Avg Mobility', 'Avg Complexity',...
    'Avg Entropy', 'Theta Entropy', 'Alpha Entropy', 'Beta Entropy'};

feature_names = {'Theta Power', 'Alpha Power', 'Beta Power', 'Theta Alpha Ratio', 'Theta Beta Ratio', 'Alpha Beta Ratio', 'Engagement Index',...
    'Theta Frontal','Theta Parietal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal', 'Alpha Occipital', 'Beta Frontal', 'Beta Temporal',...
    'Beta Parietal', 'Avg Coherence', 'Theta Coherence', 'Alpha Coherence', 'Avg Complexity',...
    'Avg Entropy', 'Theta Entropy', 'Alpha Entropy'};


%% Basic Feature Visualization

% -------------------------------------------------------------------------
% Power Spectral Density (PSD) plot
% -------------------------------------------------------------------------
mean_fft_spectrum = mean(full_fft_spectrum,3);  % Mean across all epochs
global_psd = mean(mean_fft_spectrum,1);         % Mean across all channels

f = linspace(0, fs/2, epoch_length/2);
figure(1);
plot(f,global_psd);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
title('Average PSD Across All Epochs');


% -------------------------------------------------------------------------
% General Band Power over Time (Temporal Evolution of Frequency Power of Theta, Alpha, Beta)
% -------------------------------------------------------------------------
num_epochs = size(train_features,1);
epoch_time = (1:num_epochs) * (epoch_length / fs) * (1-overlap);   % Gives Timepoints for each epoch

figure(2)
plot(epoch_time, train_features(:,1), 'b', 'LineWidth', 1.5); 
hold on;
plot(epoch_time, train_features(:,2), 'g', 'LineWidth', 1.5);
plot(epoch_time, train_features(:,3), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Band Power');
legend('Theta (4-8 Hz)', 'Alpha (8-12 Hz)', 'Beta (13-30 Hz)');
title('Frequency Band Power Over Time');
grid on;


% -------------------------------------------------------------------------
% Brain Region Specific Band Power over Time
% -------------------------------------------------------------------------
% Look for: 
% Increasing Theta Power -> might indicate higher mental workload
% Increasing Alpha Power -> might indicate more relaxation
% Increasing Beta Power -> might indicate focus or cognitive engagement

figure(3)
plot(epoch_time, train_features(:,8), 'b', 'LineWidth',1.5);
hold on
plot(epoch_time, train_features(:,9), 'g', 'LineWidth',1.5);
plot(epoch_time, train_features(:,10), 'r', 'LineWidth',1.5);
plot(epoch_time, train_features(:,11), 'c', 'LineWidth',1.5);
plot(epoch_time, train_features(:,12), 'm', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Band Power');
legend('Theta Frontal', 'Theta Temporal', 'Alpha Frontal', 'Alpha Temporal', 'Alpha Parietal');
title('Brain Region Specific Band Power Over Time');
grid on;



% -------------------------------------------------------------------------
% Engagement Index Over Time
% -------------------------------------------------------------------------
figure(4)
plot(epoch_time, train_features(:,7), 'm', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Engagement Index');
title('Engagement Index over Time');
grid on;


% -------------------------------------------------------------------------
% Coherence Heatmap
% -------------------------------------------------------------------------
figure(5);
imagesc(train_features(:,17));
colorbar;
xlabel('Channel Index');
ylabel('Channel Index');
title('Functional Connectivity (Coherence Map)');


%% Advanced Feature Visualization and Analysis 


% Investigate Feature-Label Relationship
figure;
for i = 1:9
    subplot(3,3,i);
    scatter(train_features(:,i), train_labels, 'filled');
    xlabel(sprintf('Feature %d', i));
    ylabel('Label');
    title(sprintf('Feature %d vs Label', i));
end
sgtitle('Feature vs Label Relationship');

figure;
for i = 1:13
    subplot(4,4,i);
    boxplot(train_features(:,i), train_labels);
    xlabel('MWL Label');
    ylabel(sprintf('Feature %d', i));
    title(sprintf('Feature %d Boxplot', i));
end
sgtitle('Feature Distributions by Label');


% -------------------------------------------------------------------------
% Analyzing Overall Feature Importance using Recursive Feature Elimination (RFE)
% -------------------------------------------------------------------------
% computes 10 nearest neighbors of each class (minimum 2 classes) to
% determine how much the features distinguish between classes

[ranked_features, feature_weights] = relieff(train_features, train_labels, 10); % taking the 10 nearest neighbors of features

% Sort Features by Importance
[sorted_weights, sorted_indicies] = sort(feature_weights, 'descend');

% Display features
% disp('Feature Importance Ranking:');
% for i = 1:length(sorted_indicies)
%     fprintf('%d. %s (Weight: %3f)\n', i, feature_names{sorted_indicies(i)}, sorted_weights(i));
% end

% Create bar plot
figure(6);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Create barplot with better visualization
barHandleRelieff = bar(sorted_weights, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% Add values on top of bars for better visibility
for i = 1:length(sorted_weights)
    if sorted_weights(i) >= 0
        text(i, sorted_weights(i) + 0.0002, sprintf('%.4f', sorted_weights(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(i, sorted_weights(i) - 0.0002, sprintf('%.4f', sorted_weights(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

xticks(1:length(feature_names));
xticklabels(feature_names(sorted_indicies));
xtickangle(45);
ylabel('Feature Weight', 'FontSize', 14, 'FontWeight', 'bold');
title('MWL Classification Feature Importance based on Relieff', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increase tick label font size
xlim([0, length(sorted_weights) + 1]);
ylim([min(sorted_weights)-0.0005, max(sorted_weights)+0.0005]);



% -------------------------------------------------------------------------
% Mutual Information (MI)
% -------------------------------------------------------------------------
% measures amount of shared information between a feature and the class labels
% = measures the direct feature-label correlation (model independent)
% but doesnt capture feature interactions

numFeatures = size(train_features,2);
miValues = zeros(numFeatures,1);

for i = 1:numFeatures
    miValues(i) = computeMutualInformation(train_features(:,i), train_labels, 50);
end

% % Display the MI values
% disp('Mutual Information for each feature:');
% for i=1:numFeatures
%     fprintf('%s: %.4f bits\n', feature_names{i}, miValues(i));
% end

% Visualize MI values

% % Sort MI values in descending order
[sorted_mi, sorted_indicies] = sort(miValues, 'descend');

% Bar plot
figure(7);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Create barplot with better visualization
barHandleMI = bar(sorted_mi, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% Add values on top of bars for better visibility
for i = 1:length(sorted_mi)
    if sorted_mi(i) >= 0
        text(i, sorted_mi(i) + 0.006, sprintf('%.4f', sorted_mi(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(i, sorted_mi(i) - 0.006, sprintf('%.4f', sorted_mi(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

xticks(1:length(feature_names));
xticklabels(feature_names(sorted_indicies));
xtickangle(45);
ylabel('Mutual Information (in bits)', 'FontSize', 14, 'FontWeight', 'bold');
title('MWL Classification Feature Importance based on Mutual Information (1 is highest)', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increase tick label font size
xlim([0, length(sorted_weights) + 1]);
ylim([min(sorted_weights)-0.11, max(sorted_weights)+0.11]);



% -------------------------------------------------------------------------
% Permutation Feature Importance
% -------------------------------------------------------------------------
% assesses the impact of each feature on the model's performance.
% measures how much the accuracy decreases when u randomly shuffle a
% feature's values.
% therefore is model-dependent - cannot be computed from only feature-label
% correlation (unlikey Mutual Information)

permImportance = computePermutationImportance(mdl_workload, train_features, train_labels, 10);

% Display the permutation importance for each feature
% disp('Permutation Feature Importance:');
% for i = 1:length(permImportance)
%     fprintf('%s: %.4f\n', feature_names{i}, permImportance(i));
% end

% Sort permutation importance in descending order
[sorted_permImportance, sorted_indicies] = sort(permImportance, 'descend');

% Create bar plot
figure(8);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Create barplot with better visualization
barHandlePermImportance = bar(sorted_permImportance, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% Add values on top of bars for better visibility
for i = 1:length(sorted_permImportance)
    if sorted_permImportance(i) >= 0
        text(i, sorted_permImportance(i) + 0.002, sprintf('%.3f', sorted_permImportance(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(i, sorted_permImportance(i) - 0.002, sprintf('%.3f', sorted_permImportance(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

xticks(1:length(feature_names));
xticklabels(feature_names(sorted_indicies));
xtickangle(45);
ylabel('Permutation Feature Importance', 'FontSize', 14, 'FontWeight', 'bold');
title('MWL Classification Feature Importance based on Permutation Importance', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increase tick label font size
xlim([0.5, length(sorted_permImportance) + 0.5]);
ylim([min(sorted_permImportance)-0.01, max(sorted_permImportance)+0.01]);




% -------------------------------------------------------------------------
% Visualize Channels and Brain Regions with the highest predictive power 
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Approach using Relieff
% -------------------------------------------------------------------------

% Compute Relieff - repeating this code from above in case only this section is executed
% [ranked_features, feature_weights] = relieff(train_features, train_labels, 10); % taking the 10 nearest neighbors of features

% Initialize custom topographical data vector (assuming a 14-channel EEG
% system like the Emotive EPOC device)
epoc_channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
num_electrodes = length(epoc_channels);
topo_values_alpha = nan(num_electrodes,1);

% Electrode Numbers for each Brain Region
% For STEW recording
frontal_channels = [1, 2, 3, 12, 13, 14];   % AF3 = 1; F7 = 2; F3 = 3; F4 = 12;F8 = 13; AF4 = 14;
temporal_channels = [4, 5, 10, 11];         % FC5 = 4; T7 = 5; T8 = 10; FC6 = 11;
parietal_channels = [6, 9];                 % P7 = 6; P8 = 9;
posterior_channels = [7, 8];                % O1 = 7; O2 = 8;

% Extract feature importance for each brain region
theta_frontal_weight = feature_weights(4);
alpha_frontal_weight = feature_weights(5);
alpha_temporal_weight = feature_weights(6);
alpha_parietal_weight = feature_weights(7);
alpha_posterior_weight = feature_weights(8);

% % Assign computed feature weights to the corresponding electrodes
% topo_values_theta(frontal_channels) = theta_frontal_weight;
topo_values_alpha(frontal_channels) = alpha_frontal_weight;
topo_values_alpha(temporal_channels) = alpha_temporal_weight;
topo_values_alpha(parietal_channels) = alpha_parietal_weight;
topo_values_alpha(posterior_channels) = alpha_posterior_weight;

% For the all channels / bands visualization
topo_theta = feature_weights(14:27);    % Ch1-Ch14 Theta
topo_alpha = feature_weights(28:41);    % Ch1-Ch14 Alpha
topo_beta = feature_weights(42:55);     % Ch1-Ch14 Beta


% Generate Plots to visualize the Channels:

% Define correct path for the electrode file (this is for a 10-5 64 channel electrode setup)
elp_file = fullfile(fileparts(which('eeglab.m')), 'plugins', 'dipfit', 'standard_BESA', 'Standard-10-5-cap385.elp');

% Edit locs 
locs = readlocs(elp_file);
electrode_labels = {'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'};
filtered_locs = locs(ismember({locs.labels}, electrode_labels));
writelocs(filtered_locs, '14ch_layout.locs');

% Generate Topographical Map for Alpha Power over Brain Regions
figure(10);
topoplot(topo_values_alpha, filtered_locs, 'maplimits', [min(topo_values_alpha), max(topo_values_alpha)], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Brain Region Predictive Power for MWL');

% Generate Topographical Map for Theta Power over each channel
figure(11);
topoplot(topo_theta, filtered_locs, 'maplimits', [min(topo_theta), max(topo_theta)], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Theta Band Importance / Predictive Power for MWL');

% Generate Topographical Map for Alpha Power over each channel
figure(12);
topoplot(topo_alpha, filtered_locs, 'maplimits', [min(topo_alpha), max(topo_alpha)], 'gridscale', 67, 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Alpha Band Importance / Predictive Power for MWL');

% Generate Topographical Map for Beta Power over each channel
figure(13);
topoplot(topo_beta, filtered_locs, 'maplimits', [min(topo_beta), max(topo_beta)], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Beta Band Importance / Predictive Power for MWL');

% Normalize feature importance per band to range [0,1]
topo_theta_norm = (topo_theta - min(topo_theta)) / (max(topo_theta) - min(topo_theta));
topo_alpha_norm = (topo_alpha - min(topo_alpha)) / (max(topo_alpha) - min(topo_alpha));
topo_beta_norm = (topo_beta - min(topo_beta)) / (max(topo_beta) - min(topo_beta));

% Create RGB colormap (each channel weighted by its normalized importance)
topo_combined = [topo_theta_norm, topo_alpha_norm, topo_beta_norm];

% Generate Topographical Map for all Bands over each channel
figure(14);
topoplot(topo_combined, filtered_locs, 'maplimits', [0, 1], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Combined Frequency Band Importance / Predictive Power for MWL');




% -------------------------------------------------------------------------
% Extract feature importance for channels for High and Low workload
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Approach using Permutation Importance
% -------------------------------------------------------------------------

% Compute permImportance for high and low workload separately
low_idx = train_labels == 0;
high_idx = train_labels == 1;

permImportance_Low = computePermutationImportance(mdl_workload, train_features(low_idx,:), train_labels(low_idx), 10);
permImportance_High = computePermutationImportance(mdl_workload, train_features(high_idx,:), train_labels(high_idx), 10);

% Extract the Permutation Importance for the features corresponding to the
% respective frequency bands and the workload classification
perm_theta_low = permImportance_Low(14:27);
perm_alpha_low = permImportance_Low(28:41);
perm_beta_low = permImportance_Low(42:55);
perm_total_low = perm_theta_low + perm_alpha_low + perm_beta_low;

perm_theta_high = permImportance_High(14:27);
perm_alpha_high = permImportance_High(28:41);
perm_beta_high = permImportance_High(42:55);
perm_total_high = perm_theta_high + perm_alpha_high + perm_beta_high;

figure(15);
topoplot(perm_total_low, filtered_locs, 'maplimits', [min(perm_total_low), max(perm_total_low)], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Permutation Importance of Channels to Predict Low MWL');

figure(16);
topoplot(perm_total_high, filtered_locs, 'maplimits', [min(perm_total_high), max(perm_total_high)], 'electrodes', 'on', 'emarker', {'o', 'k', 6, 3});
colorbar;
title('Permutation Importance of Channels to Predict High MWL');






%%

% -------------------------------------------------------------------------
% Statistical Analysis of Feature Importance
% -------------------------------------------------------------------------

% Computing Normality using Shapiro-Wilk/ Lilliefors/ Anderson-Darling Test, based on that use either 
% T-Test or Mann-Whitney U Test to statistically compare Feature Importance
% for Low and High Workload respectively. Then compute Cohend's d to assess
% the Feature Effect Size. 


% Initialize Result Storage
num_features = length(feature_names);

% Initializing Test Matrices to check Normality

% Kolmogorov-Smirnov Test (Simple Overview, less sensitive, compares against a normal distribution)
normality_p_values_low_kstest = NaN(num_features,1);
normality_p_values_high_kstest = NaN(num_features,1);

% Lilliefors Test (General Normality check, more sensitive, checks with unknown mean/ variance) 
normality_p_values_low_lillietest = NaN(num_features,1);
normality_p_values_high_lillietest = NaN(num_features,1);

% Anderson-Darling Test (Powerful/ Sensitive Normality check, weights tails more, which can be crucial for EEG)
normality_p_values_low_adtest = NaN(num_features,1);
normality_p_values_high_adtest = NaN(num_features,1);

% Initializing Test Matrices to check Feature Significance 
p_values_ttest = NaN(num_features,1);
p_values_mannwhitney = NaN(num_features,1);
cohens_d_values = NaN(num_features,1);

% Matrix to save selected tests
selected_tests = strings(num_features,1);

% Setup matrices for low and high MWL features respectively.
low_MWL_features = train_features(train_labels == 0, :);
high_MWL_features = train_features(train_labels == 1, :);


for i = 1:num_features

    % Extract feature values for both conditions
    low_feature_data = low_MWL_features(:,i);
    high_feature_data = high_MWL_features(:,i);

    % Check Normality: 3 possible approaches
    % Kolmogorov-Smirnov test 
    % ( test p >= 0.05 -> normally distributed; test p < 0.05 -> not normal )
    normality_p_values_low_kstest(i) = kstest((low_feature_data - mean(low_feature_data)) / std(low_feature_data));
    normality_p_values_high_kstest(i) = kstest((high_feature_data - mean(high_feature_data)) / std(high_feature_data));

    % Lilliefors test (a modified version of the Kolmongorov-Smirnov test)
    % ( test = 0 -> normally distributed; test = 1 -> not normal )
    [~, normality_p_values_low_lillietest(i)] = lillietest(low_feature_data);
    [~, normality_p_values_high_lillietest(i)] = lillietest(high_feature_data);

    % Anderson-Darling Test 
    % ( test = 0 -> normally distributed; test = 1 -> not normal )
    normality_p_values_low_adtest(i) = adtest(low_feature_data);
    normality_p_values_high_adtest(i) = adtest(high_feature_data);
    
    % Select Normality Test of choice
    % Select appropriate test based on Normality
    if normality_p_values_low_adtest(i) == 0 && normality_p_values_high_adtest(i) == 0

        % Use t-test if both distributions are normal ( p < 0.05 ->
        % Statistically significant difference; p >= 0.05 -> No significant
        % difference (feature may not be useful)
        % Statistically compare Feature Importance between High and Low
        % MWL. 
        [~, p_values_ttest(i)] = ttest2(low_feature_data, high_feature_data);   % ttest 2 = 2 sample ttest
        selected_tests(i) = "t-test";

    else

        % Use Mann-Whitney U test if at least 1 distribution is non-normal
        % (p < 0.05 -> Statistically significant difference; p >= 0.05 ->
        % No significant difference (feature may not be useful)
        % Statistically compare Feature Importance between High and Low
        % MWL. 
        p_values_mannwhitney(i) = ranksum(low_feature_data, high_feature_data);
        selected_tests(i) = "Mann-Whitney U";

    end

    % Compute Effect Size (Cohen's d)
    % 0.2 - 0.3 -> Small Effect (Weak Predictor)
    % 0.5 -> Medium Effect (Moderate Predictor)
    % 0.8+ -> Large Effect (Strong Predictor)
    mean_diff = mean(high_feature_data) - mean(low_feature_data);
    pooled_std = sqrt((std(low_feature_data)^2 + std(high_feature_data)^2) / 2);
    cohens_d_values(i) = mean_diff / pooled_std;
    
    % In the end use only features that show both statistical significance
    % AND strong effect size

end


% Display the Results in a Table
results_table = table(feature_names', normality_p_values_low_adtest, normality_p_values_high_adtest, selected_tests, ...
                      p_values_ttest, p_values_mannwhitney, cohens_d_values, ...
                      'VariableNames', {'Feature', 'Normality_p_Low_MWL', 'Normality_p_High_MWL', 'Selected_Tests', ...
                                        'T_test_p_value', 'Mann_Whitney_U_p_value', 'Cohens_d'});
disp(results_table);


% Visualize p-values in descending order
% for all features
[p_values_sorted, p_values_indicies] = sort(p_values_mannwhitney, 'descend');
sorted_p_values_features_names = feature_names(p_values_indicies);

% Only plot the significant features
significant_indicies = find(p_values_mannwhitney < 0.05);
significant_feature_names = feature_names(significant_indicies);
significant_p_values = p_values_mannwhitney(significant_indicies);

[sorted_sig_p_values, sorted_sig_indicies] = sort(significant_p_values, 'ascend');
sorted_sig_feature_names = significant_feature_names(sorted_sig_indicies);

% Create barplot
figure(17);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Create barplot with better visualization
barHandleSignificance = bar(sorted_sig_p_values, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% Add values on top of bars for better visibility
for i = 1:length(sorted_sig_p_values)
    if sorted_sig_p_values(i) >= 0
        text(i, sorted_sig_p_values(i) + 0.00002, sprintf('%.3f', sorted_sig_p_values(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(i, sorted_sig_p_values(i) - 0.00002, sprintf('%.3f', sorted_sig_p_values(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

yline(0.05, 'r--', 'LineWidth', 2); % Add red dashes line at p = 0.05

xticks(1:length(sorted_sig_p_values));
xticklabels(sorted_sig_feature_names);
xtickangle(45);
ylabel('Statistical Feature Significance', 'FontSize', 14, 'FontWeight', 'bold');
title('MWL Classification Feature Significance based on T-Test/ Mann-Whitney U-Test', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increase tick label font size
xlim([0, length(sorted_sig_p_values) + 0.5]);
ylim([min(sorted_sig_p_values), 0.055]);



% Visualization: Feature Distribution for Low vs High MWL
figure(18);
for i = 1:num_features
    subplot(4,4,i);
    hold on;
    histogram(low_MWL_features(:,i), 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    histogram(high_MWL_features(:,i), 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    title(feature_names{i});
    xlabel('Feature Value');
    ylabel('Probability');
    legend('Low MWL', 'High MWL');
    hold off;
end
sgtitle('Feature Distributions: Low vs High MWL');

% Visualization: Cohen's d 
[sorted_effect_size, sorted_cohens_indicies] = sort(abs(cohens_d_values), 'descend');
sorted_cohens_feature_names = feature_names(sorted_cohens_indicies);

% Select top 15 features and get corresponding feature names and get
% Cohen's d values
top_15_cohens_indicies = sorted_cohens_indicies(1:15);
top_15_cohens_feature_names = feature_names(top_15_cohens_indicies);
top_15_effect_sizes = sorted_effect_size(1:15);

figure(19);
bar(sorted_effect_size);
xticklabels(sorted_cohens_feature_names); % Set feature names on x-axis
xtickangle(45); % Rotate labels for readability
ylabel("Cohen's d Effect Size");
title("Feature Effect Size (Cohen's d) - Sorted by Importance");
grid on;

% Create barplot
figure(20);
set(gcf, 'Position', [100, 100, 1000, 600]);

% Create barplot with better visualization
barHandleEffectSize = bar(top_15_effect_sizes, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;

% Add values on top of bars for better visibility
for i = 1:length(top_15_effect_sizes)
    if top_15_effect_sizes(i) >= 0
        text(i, top_15_effect_sizes(i) + 0.00002, sprintf('%.3f', top_15_effect_sizes(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    else
        text(i, top_15_effect_sizes(i) - 0.00002, sprintf('%.3f', top_15_effect_sizes(i)), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
    end
end

yline(0.25, 'r--', 'LineWidth', 2); % Add red dashes line at effect size 0.25 (weak)
yline(0.5, 'r--', 'LineWidth', 2); % Add red dashes line at 0.5 (moderate)
yline(0.8, 'r--', 'LineWidth', 2); % Add red dashes line at 0.8 (strong)

xticks(1:length(top_15_effect_sizes));
xticklabels(top_15_cohens_feature_names);
xtickangle(45);
ylabel('Statistical Feature Significance', 'FontSize', 14, 'FontWeight', 'bold');
title('MWL Classification Top 15 Feature Effect Size based on Cohens d', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 14, 'LineWidth', 1.5); % Increase tick label font size
xlim([0, length(top_15_effect_sizes) + 0.5]);
ylim([min(top_15_effect_sizes), 1]);



%% Special Visualization of Feature Distribution for the Channel including Features Analysis

features_per_fig = 19;
num_figs = ceil(num_features / features_per_fig);

for fig_num = 100:100+num_figs
    figure(fig_num)

    % Determine feature range for this figure
    start_idx = (fig_num - 1) * features_per_fig + 1;
    end_idx = min(fig_num * features_per_fig, num_features);

    num_subplots = end_idx - start_idx + 1;

    for i = 1:num_subplots
        subplot(4,5,i); % 4x5 grid (20 max per figure)
        hold on;
        histogram(low_MWL_features(:,start_idx + i - 1), 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
        histogram(high_MWL_features(:,start_idx + i - 1), 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
        title(feature_names{start_idx + i - 1}, 'FontSize', 8);
        hold off;
    end

    sgtitle(sprintf('Feature Distributions (Feature %d - %d)', start_idx, end_idx));
end
