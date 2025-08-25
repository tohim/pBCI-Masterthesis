function mi = legacy_computeMutualInformation(feature, labels, numBins)

% computeMutualInformation computes the mutual information between a continuous feature and binary labels.
%
% Mutual Information (MI) is a measure of the amount of information that knowing the value of one variable 
% provides about another variable. In this case, we compute the MI between a continuous feature and binary labels.
%
% Inputs:
%   feature: A vector of continuous values (Nx1), where N is the number of samples. 
%            This represents the feature for which we want to compute mutual information.
%
%   labels: A vector of binary labels (Nx1), where each label is either 0 or 1. 
%           This represents the class or category associated with each sample in the feature vector.
%
%   numBins: (Optional) The number of bins to discretize the continuous feature. 
%            Default value is 10 if not specified. Increasing the number of bins can provide a more 
%            detailed estimate of mutual information but may also lead to sparse data in some bins.
%
% Outputs:
%   mi: A scalar value representing the computed mutual information (in units of bits) between the feature and labels.
%
% Units of bits:
% 
% The MI value is expressed in bits when using the logarithm base 2. This means that the MI quantifies the amount of information 
% in terms of binary decisions. For example, an MI of 1 bit means that knowing the feature reduces uncertainty about the label by one binary decision.
% 
% Example Interpretation
% Example 1: If mi = 0.1, this indicates a very weak relationship between the feature and the labels. 
% The feature does not provide much information about the labels.
% 
% Example 2: If mi = 0.5, this suggests a moderate relationship. 
% The feature provides some useful information about the labels, but it may not be sufficient for reliable predictions.
% 
% Example 3: If mi = 2, this indicates a (very) strong relationship. 
% The feature is highly informative, and knowing its value significantly reduces uncertainty about the labels.
%
% Example Implementation:
%   feature = randn(100, 1); % Generate 100 random continuous values
%   labels = randi([0, 1], 100, 1); % Generate 100 random binary labels
%   mi = computeMutualInformation(feature, labels, 10);
%   disp(['Mutual Information: ', num2str(mi)]);
%
% Notes:
% - The function assumes that the labels are binary (0 and 1). If the labels are not binary, 
%   an error will be raised.
% - The function handles cases where the feature is constant by slightly expanding its range 
%   to avoid division by zero during probability calculations.
% - The mutual information is calculated using the joint distribution of the feature and labels, 
%   and the marginal distributions of each variable.


% Outputs:
%   mi = computed mutual information (in bits)

if nargin < 3
    numBins = 10;   % default bins number
end

% Discretize labels (assuming binary labels, ensure they are 0 and 1)
uniqueLabels = unique(labels);

% Validate input dimensions
if length(uniqueLabels) ~= 2
    error('Labels should be binary.');
end

if length(feature) ~= length(labels)
    error('Feature and Labels must be of same length');
end

% Print Debug Info
fprintf('Feature Range: Min=%.4f, Max=%.4f, Mean=%.4f, Std=%.4f\n', min(feature), max(feature), mean(feature), std(feature));
fprintf('Labels: %d are 0, %d are 1\n', sum(labels == 0), sum(labels == 1));

% Discretize the feature into bins using manually computed edges.
% Handle the case when feature is (almost) constant
% f_min = min(feature);
% f_max = max(feature);
% 
% if f_min == f_max
%     % Expand the range slightly for constant features
%     f_min = f_min - 0.001;
%     f_max = f_max + 0.001;
% end

% Create Edges for binning the feature | Equal frequency binning
edgesX = quantile(feature, linspace(0, 1, numBins+1));
[countsX, ~] = histcounts(feature, edgesX);
pX = countsX / sum(countsX);                % Marginal Probability for the feature
pX(pX == 0) = eps;                          % Avoid log(0) 

% Since labels are binary we can compute probabilities directly
% Compute marginal probability for the binary labels:
pY = [sum(labels == 0), sum(labels == 1)] / length(labels);

% Compute joint distribution: use histcounts2 (bivariate histogram bin counts)
% For labels we use 2 bins corresponding to the unique values
edgesY = [-0.5, 0.5, 1.5];                  % Define edges for binary labels
[jointCounts, ~, ~] = histcounts2(feature, labels, edgesX, edgesY);
pXY = jointCounts / sum(jointCounts(:));    % Joint probability distribution
pXY(pXY == 0) = eps;                        % Avoid log(0)

% % Ensure pX and pY are column vectors of same size as pXY
% pX = pX(:);
% pY = pY(:);

% Expand pX and pY to match pXY dimensions
pX_mat = repmat(pX,1,2);                    % numBins x 2
pY_mat = repmat(pY', numBins, 1);           % numBins x 2

% Probability Value Checks before Log Computation
disp('pX values (marginal feature distribution):');
disp(pX');
disp('pY values (label distribution):');
disp(pY);
disp('pXY joint distribution:');
disp(pXY);

% Calculate Mutual Information
mi = sum(pXY(:) .* log2(pXY(:) ./ (pX_mat(:) .* pY_mat(:))), 'omitnan');

fprintf('Mutual Information: %.4f bits\n', mi);
end

