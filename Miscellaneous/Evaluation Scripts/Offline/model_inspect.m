% Checking Models 
clc; clear

%%
% EASY DIFF
% Load model and inspect
model_data = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_diff_model.mat');
who('-file', 'hyper_1000_25wCsp_4sec_proc5_MATB_easy_diff_model.mat')

% Check Label Distribution
Y = model_data.mdl.Y;
disp(unique(Y));  % Should be [0 1]
disp(mean(Y));    % Should reflect class balance (e.g., ~0.5)

% Quick Check decision boundary orientation (direction the model is
% leaning)
[~, score] = predict(model_data.mdl, model_data.mdl.X);
histogram(score(:,2)); title('Decision Scores for Class 1');

% Manually evaluate the model on known test data
test_data = load('1000_25wCsp_4sec_proc5_MATB_easy_diff_test_features.mat');
test_labels = load('1000_4sec_MATB_easy_diff_test_labels.mat');

[predictions, scores] = predict(model_data.mdl, test_data.test_features);
confusionmat(test_labels.test_labels, predictions)

%%
% EASY MED DIFF
% Load model and inspect
model_data = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat');
who('-file', 'hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat')

% Check Label Distribution
Y = model_data.mdl.Y;
disp(unique(Y));  % Should be [0 1]
disp(mean(Y));    % Should reflect class balance (e.g., ~0.5)

% Quick Check decision boundary orientation (direction the model is
% leaning)
[~, score] = predict(model_data.mdl, model_data.mdl.X);
histogram(score(:,2)); title('Decision Scores for Class 1');

% Manually evaluate the model on known test data
test_data = load('1000_25wCsp_4sec_proc5_MATB_easy_meddiff_test_features.mat');
test_labels = load('1000_4sec_MATB_easy_meddiff_test_labels.mat');

[predictions, scores] = predict(model_data.mdl, test_data.test_features);
confusionmat(test_labels.test_labels, predictions)


%%
% Compare MATBs
mdl1 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_diff_model.mat');
mdl2 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat');

X1 = mdl1.mdl.X;
X2 = mdl2.mdl.X;

% Predict model 1's training data with model 2
[~, scores1_on_2] = predict(mdl2.mdl, X1);
[~, scores2_on_1] = predict(mdl1.mdl, X2);

% Are the scores inverted?
corrcoef(scores1_on_2(:,2), -scores2_on_1(:,2))  % should be strongly positive if inverse



%% Compare predictions of both models on same input

mdl1 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_diff_model.mat');
mdl2 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat');

X_test_features = load('1000_25wCsp_4sec_proc5_STEW_test_features.mat');

pred1 = predict(mdl1.mdl, X_test_features.test_features);
pred2 = predict(mdl2.mdl, X_test_features.test_features);


% Check for exact inversion
flipped = sum(pred1 == (1 - pred2)) / length(pred1);
identical = sum(pred1 == pred2) / length(pred1);

fprintf('Flipped: %.2f %% | Identical: %.2f %%\n', 100*flipped, 100*identical);


% Compare support vectors and weights
disp(dot(mdl1.mdl.Beta, mdl2.mdl.Beta));


%% Final conclusion

% Result: 
% The two models do not produce perfectly inverted predictions
% ~70% of the predictions are identical
% ~30% are flipped — this is not trivial, but it’s not systematic inversion either
% So, your original observation that “they act like mirrored models” was not literally true, but 
% it still reflects a meaningful behavioral difference between them.
% dot = 0: This indicates:
% The SVM weight vectors (i.e., mdl1.mdl.Beta and mdl2.mdl.Beta) are orthogonal
% The decision hyperplanes are not aligned, and their orientations are uncorrelated
% If they were:
% Similar → dot ≈ 1
% Inverted → dot ≈ -1
% Unrelated/orthogonal → dot ≈ 0
% This tells us that the two models:
% Are not learning the same decision boundary, even though the datasets share the same task
% Likely learned to separate the classes in different directions in feature space due to the inclusion of 
% med trials in the meddiff variant
% --> 
% The models are not inverted.
% But they are meaningfully different — orthogonal boundaries, ~30% disagreement.
% This explains why classwise prediction counts and metrics appear "mirrored", even though total accuracy is the same
% ItIs actually a case of: Boundary shift + label distribution shift → semantic drift


%% Which model is now better at predicting High MWL? 

mdl1 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_diff_model.mat');
mdl2 = load('hyper_1000_25wCsp_4sec_proc5_MATB_easy_meddiff_model.mat');

X_test_features = load('1000_25wCsp_4sec_proc5_STEW_test_features.mat');
X_test_labels = load('1000_4sec_STEW_test_labels.mat');

[pred1, score1] = predict(mdl1.mdl, X_test_features.test_features);
[pred2, score2] = predict(mdl2.mdl, X_test_features.test_features);

fprintf('MATB EASY DIFF\n');
compute_high_mwl_metrics(pred1, score1, X_test_labels.test_labels);
fprintf('MATB EASY MED-DIFF\n');
compute_high_mwl_metrics(pred2, score2, X_test_labels.test_labels);


function compute_high_mwl_metrics(predictions, scores, true_labels)
    % Scores = Nx2 matrix, second column = confidence for class 1

    TP = sum((predictions == 1) & (true_labels == 1));
    FP = sum((predictions == 1) & (true_labels == 0));
    FN = sum((predictions == 0) & (true_labels == 1));

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * (precision * recall) / (precision + recall);

    fprintf('Precision (High MWL): %.2f %%\n', 100*precision);
    fprintf('Recall    (High MWL): %.2f %%\n', 100*recall);
    fprintf('F1-Score  (High MWL): %.2f %%\n', 100*f1);

    % ROC and AUC
    [~, ~, ~, AUC] = perfcurve(true_labels, scores(:,2), 1);
    fprintf('AUC       (High MWL): %.2f\n\n', AUC);
end


% Interpretation:

% The model trained on MATB_easy_meddiff:
% Is better at identifying true high MWL cases (higher recall),
% Is more accurate when it predicts high MWL (higher precision),
% Achieves a higher overall F1 score, indicating a better balance between false alarms and missed detections.
% 
% Especially Important:
% Recall improved from 44% → 57.3%, which is significant — your model detects ~30% more actual high workload cases with easy_meddiff.
% 
% Conclusion
% The easy_meddiff variant is objectively better at detecting High MWL, and should be your preferred base for MATB-based models — especially for adaptive control or safety-critical decisions.

