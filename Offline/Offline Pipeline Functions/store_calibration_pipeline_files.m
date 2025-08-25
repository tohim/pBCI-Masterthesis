function store_calibration_pipeline_files(params, datasets)

fprintf('\n[STAGE X] Saving Calibration Pipeline Metadata and Files...\n');

% Base Paths
base_path = fullfile('E:', 'SchuleJobAusbildung', 'HTW', 'MasterThesis', ...
                     'Code', 'Matlab', 'Data', 'AutoCalibration', 'v1');        % !! CHANGE VERSION FOLDER IF NEEDED !!
samples_main_name  = sprintf('%dsamples_%dPct_Calib', params.total_samples, params.calib_ratio*100);
model_data_name    = sprintf('%dsamples_calib_models', params.total_samples);

main_folder        = fullfile(base_path, samples_main_name);
model_folder       = fullfile(base_path, model_data_name);
figures_folder     = fullfile(main_folder, 'figures');
final_model_folder = fullfile(main_folder, model_data_name);

% Create needed folders
if ~exist(main_folder, 'dir'), mkdir(main_folder); end
if ~exist(model_folder, 'dir'), mkdir(model_folder); end
if ~exist(figures_folder, 'dir'), mkdir(figures_folder); end

% Move model folder into main output folder if not yet moved
if exist(model_folder, 'dir') && ~exist(final_model_folder, 'dir')
    movefile(model_folder, final_model_folder);
    fprintf('Moved model folder into: %s\n', main_folder);
end

% -------------------------------------------------------------------------
% Save all figures
% -------------------------------------------------------------------------
figs = findall(0, 'Type', 'figure');
for i = 1:numel(figs)
    fig = figs(i);
    figure(fig);
    ax = get(fig, 'CurrentAxes');
    title_text = '';
    if ~isempty(ax)
        title_obj = get(ax, 'Title');
        if isprop(title_obj, 'String') && ~isempty(title_obj.String)
            title_text = title_obj.String;
            if iscell(title_text), title_text = strjoin(title_text); end
        end
    end
    if ~isempty(title_text)
        clean_title = regexprep(title_text, '[^\w\s-]', '');
        clean_title = strtrim(regexprep(clean_title, '\s+', '_'));
        filename_base = sprintf('fig_%d_%s', i, clean_title);
    else
        filename_base = sprintf('figure_%d', i);
    end

    savefig(fig, fullfile(figures_folder, [filename_base '.fig']));
    saveas(fig, fullfile(figures_folder, [filename_base '.png']));
end

% -------------------------------------------------------------------------
% Move .mat Files (Models) into: [source_dataset]/[cross_dataset]/model.mat
% -------------------------------------------------------------------------
mat_files = dir('*.mat');
for k = 1:length(mat_files)
    fname = mat_files(k).name;

    % Handle .mat files not related to specific model storage
    if strcmp(fname, sprintf('AfterCalib_%dTotal_%dCalib_Results.mat', params.total_samples, params.calib_samples))
        movefile(fname, fullfile(main_folder, fname));
        continue;
    elseif ismember(fname, {'PostCalib_Cross_Results.mat', 'PreCalib_Cross_Results.mat'})
        continue;  % handled later
    end

    % Try to parse: source and cross dataset from the filename
    source_found = false;
    for s = 1:numel(datasets)
        src = datasets{s};
        if contains(fname, ['_' src '_'])  % Source dataset detected in filename
            for c = 1:numel(datasets)
                cross = datasets{c};
                if strcmp(src, cross), continue; end % Skip self-pairing
                if contains(fname, ['wCross_' cross])
                    % Matched: src → cross
                    target_folder = fullfile(final_model_folder, src, cross);
                    if ~exist(target_folder, 'dir')
                        mkdir(target_folder);
                    end
                    movefile(fname, fullfile(target_folder, fname));
                    source_found = true;
                    break;
                end
            end
        end
        if source_found, break; end
    end

    % If not a cross-model (e.g., pre-calibration or metadata), put in base folder
    if ~source_found
        movefile(fname, fullfile(final_model_folder, fname));
    end
end

% -------------------------------------------------------------------------
% Move Excel result and stats files
% -------------------------------------------------------------------------
results_xlsx = sprintf('AfterCalib_%dTotal_%dCalib_Results.xlsx', ...
                       params.total_samples, params.calib_samples);
stats_xlsx   = sprintf('AfterCalib_%dTotal_%dCalib_Stats.xlsx', ...
                       params.total_samples, params.calib_samples);

movefile(results_xlsx, fullfile(main_folder, results_xlsx));
movefile(stats_xlsx,   fullfile(main_folder, stats_xlsx));

% -------------------------------------------------------------------------
% Move other result files (logs, delta, etc.)
% -------------------------------------------------------------------------
extra_patterns = {
    'Calibration_Delta_Results.xlsx'
    'PostCalib_Cross_Results.mat'
    'PreCalib_Cross_Results.mat'
    'log_*.txt'
};

for p = 1:numel(extra_patterns)
    matches = dir(extra_patterns{p});
    for m = 1:length(matches)
        movefile(matches(m).name, fullfile(main_folder, matches(m).name));
        fprintf('Moved: %s\n', matches(m).name);
    end
end

fprintf('All calibration pipeline files stored in: %s\n', main_folder);
close all;
end
