## Real-Time and Online passive Brain-Computer Interface Mental Workload Monitoring with Adaptive Automation - Code and Data Repository


<br>


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Abstract
This thesis designs and evaluates a passive Brain-Computer Interface (pBCI) for monitoring mental workload (MWL) in a factory-like human–robot setting, by triggering adaptive automation (AA) to improve neuroergonomic 
outcomes. A 14-channel EEG pipeline streams, preprocesses, and extracts multi-domain features (band-power/ratios, coherence, hjorth, entropy, CSP). Support vector machines (SVMs) trained on public MWL datasets are 
calibrated per subject and combined with majority-vote smoothing. The end-to-end latency is < 2 seconds per epoch, enabling real-time operation. Offline, within-dataset classification achieves ~80% accuracy (best single model 84.67%), 
validating the feature–classifier choice. In the experiment, calibration of the base models increases cross-data accuracy by +7–16%, a real-time accuracy of 56.59% was achieved and the online decision smoothing reaches a block-level 
adaptation accuracy of 66.67%. Subjective workload assessment (NASA-TLX) confirms distinct LOW/HIGH states and indicates effective neuroadaptive automation: under pBCI control, perceived workload in the HIGH condition is significantly 
reduced and the HIGH–LOW gap narrows. The LOW condition shows no reliable change, suggesting that balancing is achieved primarily by relieving overload. Overall, the results demonstrate the technical feasibility of real-time EEG-based 
MWL monitoring and subsequent workload mitigation, while highlighting calibration and decision smoothing as key enablers.

Keywords: passive Brain-Computer Interface (pBCI), electroencephalography (EEG), real-time classification, mental workload, adaptive automation (AA), neuroergonomics, NASA-TLX, human–robot interaction (HRI), support vector machine (SVM).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

<br>
<br>


## Repository Structure


1. **Data**
   
   - The Offline and Real-Time Calibration and Experiment Data is available in the "Releases": https://github.com/tohim/pBCI-Masterthesis/releases
   - Select a Data Release, open "Assets", download the .zip file that contains all the respective data.

2. **Code**  
   - Code-folders containing the respective scripts, functions, data, etc. for:
   - Real-Time Code, Offline Code and Miscellaneous Code
  
3. **Citation**

   <br>
   <br>
   <br>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------   

## 1. Data

<br>
   
### Real-Time Data:

   The real-time data is structured into 2 major phases of the whole measurement:

   - **Calibration phase**: a total of 20 blocks, 10 low/10 high MWL conditions, with 60 seconds per block. Each block contains 30 epochs, one epoch is 4 seconds, epochs are buffered from the continuous EEG stream with 50% overlap.
   Each individual 4 second epoch is a standalone MWL EEG segment, no specific reversal of the 50% overlap is necessary to work with the data.
   The last 8 blocks (4 low/4 high) of the 20 calibration blocks are used for the subject-specific calibration phase before the experiment phase. All other data was recorded for subsequent research on pBCI and MWL estimation following this thesis.

   - **Experiment Phase**: a total of 10 blocks, 5 low/5 high MWL conditions, 60 sec per block, each block contains 30 epochs, 4 second epochs, buffered with 50%, identical to the calibration phase.
   During the experiment phase, adaptive automation (AA) is performed after 20 epochs/ 40 seconds. The system changes behavior based on the predicted MWL of the subject. Therefore, experiment data after the AA was performed is not
   aligning anymore with the present ground truth of an experiment block. When working with the experiment data, specifically select pre-AA epochs, to target EEG data and derived features corresponding to clear, binary ground truth MWL (0 = low, 1 = high)

  > [!IMPORTANT]
  > Both the Calibration and pre-adapt Experiment Data can be used for subsequent research on binary (low/high) MWL states.
  
   <br>
   <br>

  **Real-Time Data stored in the Releases**:
   
  - RT_Experiment_Data: Real-Time Experiment Data recorded in this study (contains measurement information and all results)
  - RT_Calibration_Data: Real-Time Calibration Data recorded in this study (contains measurement information and all results)
  - RT_Base_Models: Contains the 3 Base Models used for the MWL Classification in the Experiment + the respective dataset-specific CSP filters. The 3 Base Models are also found in the code-folder "Real Time" -> "BaseModels"
  - RT_Calibrated_Models: Contains all subject-specific calibrated models (Base models finetuned/ retrained on the RT_Calibration_Data) + the same dataset-specific CSP filters as for the Base Models (CSP filters do not get retrained)

  <br>
  <br>
  
  **How to access the RT Data**:
       
  - Download the .zip files, extract the zip, access the subject-specific .mat files from both the RT_Calibration_Data & the RT_Experiment_Data. (contains the 4-second EEG epochs, together with additional metrics and information of each trial)
  - Within RT_Calibration_Data & RT_Experiment_Data: .mat files with the measurement logs/data + .txt summary containing extra measurement information for each subject and respective phase
  - Both the calibration_log.mat (saved in the Subject*_CalibrationLog.mat file) and the experiment_log.mat (saved in the Subject*_Results.mat file) are "structs":
    - a struct array is a Matlab data container allowing to save whole matrices (e.g., the channels x samples EEG data, in row x column format) within a "Field" of the struct
    - to access the respective EEG data matrix, target the respective field (struct.field(position); e.g., experiment_log.processed(5) -> gives the processed EEG matrix for field/ struct-row number 5)
        
  <br>
  <br>
  
  **Naming Convention/ Data Structure**:
         
  - Pre-downsampled Raw ("*full_raw*"): 250 Hz Sampling Rate (fully raw data from the continuous EEG stream via TCP/IP port coming from Simulink (check the "acquire_new_data" function in "Real Time" -> "Functions" folder for further information)
  - Downsampled Raw ("*raw*"): 128 Hz Sampling Rate (raw data downsampled to the 128 Hz, aligning with the training data of the public MWL datasets (check the "acquire_new_data" function in "Real Time" -> "Functions" folder for further information)
  - Reordered ("*raw_reordered*"): Changing Nautilus Channel Numeration reordered to the Emotiv Epoc Channel Numeration (check the "order_channels" function in "Real Time" -> "Functions" folder for further information)
  - Processed ("*processed*"): Preprocessed EEG data (Filtering, Artifact Removal, (check the "RT_preprocess_epochs" function in "Real Time" -> "Functions" folder for further information)

<br>
<br>

  **EEG Data**:
  
**Subject#_CalibrationLog.mat** -> "calibration_log": contains the pre-downsampled-raw/ downsampled-raw/ reordered/ processed 4-second EEG segments of the real-time calibration phase

  > [!IMPORTANT]
  > For calibration data, always use the **last 600** epochs for subsequent research.
      <br>
  
  Only the LAST 600 epochs stored within the Calibration Logs are relevant data. Select accordingly when working with the data. All data before are recordings from familiarization periods in the
  beginning of each trial. These were performed to allow each subject to gain some initial experience with the paradigm, task and specifically working together with the robot. 

  <br>
  <br>

**Subject#_Results.mat** -> "experiment_log": contains the pre-downsampled-raw/ downsampled-raw/ reordered/ processed 4-second EEG segments of the real-time experiment phase

  > [!IMPORTANT]
  > For experiment data, use **pre-ADAPT** epochs for clear ground truth: **first 19 epochs** of each block (total **190/300** epochs).
      <br>

  Only the FIRST 19 epochs of each ground truth block are PRE-ADAPT epochs. Filtering for pre-adapt epochs results a total of 190/300 epochs stored within the Experiment Log of each subject.
  Afterwards (after 19/30 epochs per block), the "Adaptive Automation" (AA) occurs, the MWL state of the subject changes and no clear binary MWL classification is given anymore. 
  It is necessary to sort the Experiment Log data for pre-adapt epochs, if clear binary 0=low/ 1=high data is wanted. This can be done be utilizing the logged "adapted_epochs" in the experiment_log struct.
        
  Example:
    
  ```matlab
% Get all indices of epochs before applying "ADAPT":
pred_idxs = isnan([experiment_log.adapted_epochs]);

% Extract all pre-adaptation processed EEG:
pre_adapt_processed_EEG = cell2mat({experiment_log(pred_idxs).processed}');

% Extract all pre-adaptation STEW features:
pre_adapt_features_STEW = cell2mat({experiment_log(pred_idxs).STEW_features}');
  ```
    
<br> 
<br> 

 **Both the Calibration and Experiment Logs also contain**:
         
  - Dataset/Model-specific Features:
    
    In calibration_log: *features_STEW/HEAT/MATB*  |  In experiment_log: *STEW/HEAT/MATB_features*

  > [!IMPORTANT]
  > For calibration & experiment data: First 25 features are identical across datasets. The last 6 are dataset-specific CSP-filter derived features.
      <br>
             
  The first 25 (of 31 total) features (for each dataset-feature-vector) are identical across all 3 datasets.
               The last 6 features of each dataset-feature-vector are the dataset-specific CSP Features.
               For each dataset, a specific CSP-filter is derived/ trained -> resulting in dataset-specific CSP features. All other features are derived from the the same feature extraction and do not differ (check "RT_extract_features" function
               for the (25) Base Features and "extract_csp_features_single_epoch" for CSP Features extraction in the "Real Time" -> "Functions" folder for further information)

  Example:
    
  ```matlab

  %% Extracting Labels and Condition-Specific Features + Combining all individual Features

  % Get all indices of epochs before applying "ADAPT":
  pred_idxs = isnan([experiment_log.adapted_epochs]);

  % True labels before applying "ADAPT"
  true_label = cell2mat({experiment_log(pred_idxs).true_label})';

  % Extract all pre-adaptation Features
  all_epoch_features_STEW = cell2mat({experiment_log(pred_idxs).STEW_features}');
  all_epoch_features_HEAT = cell2mat({experiment_log(pred_idxs).HEAT_features}');
  all_epoch_features_MATB = cell2mat({experiment_log(pred_idxs).MATB_features}');

  % Get indicies to sort after labels
  low_idx  = true_label == 0;
  high_idx = true_label == 1;

  % Combine respective low and high epoch Features
  base_low  = all_epoch_features_STEW(low_idx, 1:25);  % take any dataset (1:25, to target the first 25 (low) Base-only feature set)
  base_high = all_epoch_features_STEW(high_idx, 1:25); % same for high

  stew_low  = all_epoch_features_STEW(low_idx, 26:31); % take specifically the last 6 STEW CSP features (26:31)
  stew_high = all_epoch_features_STEW(high_idx, 26:31);

  heat_low  = all_epoch_features_HEAT(low_idx, 26:31); % take specifically the last 6 HEAT CSP features
  heat_high = all_epoch_features_HEAT(high_idx, 26:31);

  matb_low  = all_epoch_features_MATB(low_idx, 26:31); % take specifically the last 6 MATB CSP features
  matb_high = all_epoch_features_MATB(high_idx, 26:31);

  % Combine the base with the dataset-specific features to get total features across epochs
  combined_low  = [base_low, stew_low, heat_low, matb_low];
  combined_high = [base_high, stew_high, heat_high, matb_high];
  ```
    
<br> 

  - **Additional Columns of the respective structs: General Measurement and Paradigm Information**
             
    - Calibration_Log:
      
        - *block* (block number of the calibration phase)
          
        - *Sample/Epoch count per Block* (individual epoch count within each block)
          
        - *true_label* (low=0, high=1 MWL condition)
          
        - *timestamp* (information about Date and Time of the measurement, can also be seen in the CalibrationSummary.txt alongside other details)
          
                 
    - Calibration_Log_Metadata: saved and reused *rng_seed* for block order/ sequence shuffling + respective *block_labels* sequence

      <br>    
  
    - Experiment_Log:
        
        - *true_label* (low=0, high=1 MWL condition)
          
        - *predicted_MWL_stew/heat/matb* (real-time individual model predictions)
          
        - *majority_MWL* (NaN if not valid, 0 (predicts LOW) or 1 (predicts HIGH) majority-vote predictions)
          
        - *adapt_command* (empty string, or the respective block-level prediction/ adaptation command of the pBCI system)
          
        - *correct* (Nan if no majority-vote predictions, 1 correct or 0 if false (majority vote vs. true_label))
          
        - *adapted_epochs* (NaN for pre-adaptation epochs of each block, current block number of post-adaptation epochs of each block)
          
        - *STEW/HEAT/MATB_classifier_confidence* (saves model specific confidence scores for each prediction)
          
        - *epoch_start* and *epoch_end* (saving total epoch processing duration)
             
    - For the Experiment_Log a Block-Numeration is missing. It can be added by up-counting block numbers after each 30th block (1 block is 30 epochs). The epoch number is identical to the "Fields" count of the struct.
                 
    - Experiment_Log_Metadata: saved and reused *rng_seed* for block order/ sequence shuffling + respective *block_labels* sequence
      
   <br> 
   
   -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
  
### Offline Data:

  <br>

  **Public MWL Datasets**:

  - The 3 public MWL datasets used in this study to train the Base Models are available here:
    -  STEW: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
       
         STEW study: https://ieeexplore.ieee.org/document/8478165
    
    -  HEAT: https://ddd.uab.cat/record/259591
           
         HEAT study: https://www.mdpi.com/1424-8220/24/4/1174
    
    -  MATB: https://zenodo.org/records/4917218 (dataset only including the MATB-II task recordings)
           
         MATB (COG-BCI) study: https://www.nature.com/articles/s41597-022-01898-y#ref-CR21

         Full COG-BCI dataset : https://zenodo.org/records/7413650 (including all tasks)
       
    <br>
    <br>

     **Thesis Offline Data**:

     **To access and download the stored Offline Data, again check the Releases for the respective "Offline_*" Tags:**
    <br>
    <br>

     - Offline_*datasetname*: Offline Segmented, Labeled, Processed STEW/ MATB Easy-Diff/ MATB Easy-MedDiff/ HEAT Data (contains the respective 4-second epoch labels and segmented raw & processed data of each public MWL dataset)

       - Sample Size (e.g., 1000) -> Dataset (e.g., STEW) -> "1000_4sec_STEW_sampled_labels.mat": Labels used for the Base Models and also later for the Calibration (for feature set splitting into train, val, test) -> resulting
         in respective "4sec_train_labels.mat", "4sec_val_labels.mat", "4sec_test_labels.mat". The respective labels are already available and stored in the following "Offline_Features&Models" Release/Tag.

       - Naming Convention: "4sec _ processingType _ datasetName"
         
         - *processingType* = raw, proc5
           
         - *datasetName* = STEW, HEAT, MATB_easy_diff, MATB_easy_meddiff
           
         - For more detailed explanation of the meaning of each part of the naming, check the "OFF_pipeline.mat" "File Naming Convention" at the top of the script
         
     <br>
     <br>
     
     - Offline_Features&Models: Offline Data of both the pre- and post-calibration "OFF_pipeline" script. The "Offline_Data.zip" is available, containing all the respective offline data.
    
         AutoPipeline Folder:
    
          - Total amount of Base Models trained and evaluated in this study.
         
             - Naming Convention: "#trainingSamples _ featureConfiguration _ 4sec _ preprocessingType _ datasetName _ modelType"
       
               - *trainingSamples* = 1000, 2000, 3000, 4000
           
               - *featureConfiguration* = 25 (BASE-only), csp (CSP-only), 25wCsp (BASE+CSP)
           
               - *processingType* = proc5
           
               - *datasetName* = STEW, HEAT, MATB_easy_diff, MATB_easy_meddiff
           
               - *modelType* = standard, hyper, norm, hypernorm
           
               - For more detailed explanation of the meaning of each part of the naming, check the "OFF_pipeline.mat" "File Naming Convention" at the top of the script
              
               <br>
               
         AutoCalibration Folder:
        
          - Offline Calibrated Models: selected Base Models calibrated w other Base Model data (e.g., STEW calibrated with HEAT)
            
            - Naming Convention: "modelType _ #trainingSamples _ featureConfiguration _ 4sec _ preprocessingType _ datasetName _ calibrationType _ wCross _ crossDataset"
         
              - *modelType* = standard, hyper, norm, hypernorm
           
              - *trainingSamples* = 1000, 2000, 3000, 4000
           
              - *featureConfiguration* = 25 (BASE-only), csp (CSP-only), 25wCsp (BASE+CSP)
           
              - *processingType* = proc5;
           
              - *datasetName* = STEW, HEAT, MATB_easy_diff, MATB_easy_meddiff
           
              - *calibrationType* = adapted_norm, finetuned, finetuned_adapted_norm
           
              - *crossDataset* = STEW, HEAT, MATB_easy_diff, MATB_easy_meddiff
           
              - For more detailed explanation of the meaning of each part of the naming, check the "OFF_pipeline.mat" "File Naming Convention" at the top of the script

<br>

-----------------------------------------------------------------------------------------------------------------------------------------------

<br>

## 2. Code

   <br>
   
### Real-Time Code:


  Code-folder containing all necessary scripts, functions, Simulink model, data (models, CSP filters) and information (Paradigm.txt, Setup_Check.txt) to perform real-time measurement.
   
  - BaseModels: 3 base models used for the experiment (STEW, HEAT, MATB)
      
  - Functions: Matlab functions used for to execute the "RT_pipeline"
      
  - Robot: Python scripts to send movement commands to robot
      
  - gNautilus_model_LSL_32_Thomas_22a.slx: Simulink model to control continuous EEG streaming via TCP/IP
      
  - Paradigm: Info Sheet Paradigm
      
  - RT_pipeline: Full Real-Time Experiment (and Calibration) Paradigm Code (TCP/IP handling, catching continuous EEG stream, preprocessing, feature extraction, MWL classification, command sending, screen cue handling,
      measurement info and data logging)
   
  - Setup_Check: Measurement instructions

   <br>
   <br>
  
### Offline Code:


  Code-folder containing all necessary scripts, functions, data (labels, features, models) and information (comments in the code) to perform offline analysis of the public MWL datasets, respective features, models and the calibration phase + respective statistical analysis.
         
  - Offline Pipeline Functions: Matlab functions used to execute the "OFF_pipeline"
      
  - Segmentation Scripts: Matlab scripts to segment the STEW/ HEAT/ MATB datasets into random shuffled 4-second LOW and HIGH epochs (the "OFF_pipeline" 
      automation script uses the segmented and preprocessed data of each dataset.
   
  - OFF_pipeline: Extensive automation script to extract features, train, evaluate and hyperparameter-tune the base models, test and evaluation of 
      features, includes pre-calibration cross-data check, includes complete calibration phase (all 3 calibration approaches, automatic source -> target dataset 
      handling and model retraining together with all post-calibration within and cross-data evaluations and respective excel sheet generation)

  - Helpers: Helper functions

   <br>
   <br>
  
### Miscellaneous Code:


Code-folder containing parts of the scripts and functions to perform post-analysis or summaries of the offline results and real-time experiment results (evaluation across all subjects: signal quality, feature statistics and importance, model performances, NASA-TLX answers, etc.). Also contains old legacy scripts as reference for newer versions.

   - Evaluation Scripts: Scripts used to for offline and experiment data evaluations
      
   - Legacy Functions: Old functions, not used anywhere anymore, included for reference
    
<br>
<br>
<br>
<br>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 3. Citation


T. Himmelstoß, “Implementation and Assessment of a Real-Time Passive Brain–Computer Interface for Neuroadaptive Automation in Factory-like Settings”,
M.Sc. Thesis, Systems Neuroscience & Neurotechnology Unit, Faculty of Engineering, Saarland Univ. of Applied Sciences (htw saar), Saarbrücken, Germany, 2025.
