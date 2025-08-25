function permImportance = legacy_computePermutationImportance(model, features, labels, numPermutations)

% Computes permutation feature importance for a trained model.

% Inputs:
% model: trained classification model
% features: feature matrix (samples x features)
% labels: true labels for the samples (samples x labels)
% numPermutations: number of permutations per feature (default: 10)

% Output:
% permImportance: vector containing the importance scores for each feature

if nargin < 4
    numPermutations = 10;
end

% Compute baseline accuracy
baselinePred = predict(model, features);
baselineAcc = sum(baselinePred == labels) / length(labels);
numFeatures = size(features,2);
permImportance = zeros(numFeatures,1);

% Loop over each feature
for i = 1:numFeatures
    accPerm = zeros(numPermutations,1);
    for j = 1:numPermutations
        % Create a copy of the features and permute the i-th feature
        featuresPerm = features;
        featuresPerm(:,i) = featuresPerm(randperm(size(features,1)), i);
        predPerm = predict(model, featuresPerm);
        accPerm(j) = sum(predPerm == labels) / length(labels);
    end
    % Importance is the average drop in accuracy
    permImportance(i) = baselineAcc - mean(accPerm);
end


end

