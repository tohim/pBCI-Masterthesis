function cleaned = remove_artifacts(epoch_data, fs, window_sec, overlap, threshold, corr_threshold)

% remove_artifacts - Cleans EEG epoch using MAD-based artifact detection,
% linear interpolation, fallback neighbor-window interpolation, and
% inter-channel correlation-based channel-level correction.
%
% Inputs:
%   epoch_data     - [channels x time] EEG segment (single epoch)
%   fs             - Sampling frequency (Hz)
%   window_sec     - Sliding window length in seconds (e.g., 1 sec - to
%                    cover an eyeblink for sure (300-700 ms artifacts) - also keeps moderate
%                    local precision to avoid risk of over-smoothing valid EEG across longer stretches)
%   overlap        - Overlap ratio (e.g., 0.75 = 25% overlap) - with overlap of 75% for 1 sec epoch 
%                    we get windows of 750 ms - which should be sufficient so smooth out eye blinks)
%                    overlap values:
%                    0.25 Step = 25% of win	-> 75% overlap
%                    0.5 Step  = 50% of win	-> 50% overlap
%                    0.75 Step = 75% of win	-> 25% overlap
%   threshold      - MAD threshold (e.g., usually 6-8) - i chose 5, as it showed to be necessary for some outliers
%   corr_threshold - Channel correlation rejection threshold (e.g., 0.5) -
%                    assuming that usually eeg channels have r = 0.6-0.95 across frontal/
%                    central/ occipital channels & r = 0.1-0.3 for artifacted channels
%
% Output:
%   cleaned        - Cleaned EEG data of same size as input


[num_ch, num_pts] = size(epoch_data);
win_length = round(window_sec * fs);
step = round(win_length * overlap);  % 50% overlap - balanced temporal resolution of artifact removal 
                                     % (increased chance to have at least 1  with enough clean samples)
             
cleaned = epoch_data;                % Initialize cleaned output

% Determine number of windows covering signal
num_windows = floor((num_pts - win_length) / step) + 1;
final_window_start = (num_windows - 1) * step + 1;

% Check if last part of the epoch would be missed → Add 1 final window
if final_window_start + win_length - 1 < num_pts
    num_windows = num_windows + 1;
end

% Prepare memory for storing cleaned window segments
window_data = zeros(num_ch, win_length, num_windows);

% Minimum good samples to accept direct interpolation
min_good_samples = round(0.1 * win_length);  % 10% of samples per window

% Sanity Check for Interpolated Segments
max_amp = 180;      % Max acceptable absolute amplitude in µV
max_diff = 250;     % Max jump between consecutive samples in µV


% ========== MAD-based artifact cleaning =============
for ch = 1:num_ch
    for w = 1:num_windows
        idx_start = (w - 1) * step + 1;
        idx_end = min(idx_start + win_length - 1, num_pts);  % Avoid overrun
        segment = epoch_data(ch, idx_start:idx_end);

        % Compute robust range via MAD
        med = median(segment);
        mad_val = mad(segment, 1);
        lower = med - threshold * mad_val;
        upper = med + threshold * mad_val;

        % Detect artifact indices
        artifact_idx = find(segment < lower | segment > upper);
        good_idx = setdiff(1:length(segment), artifact_idx);

        if numel(good_idx) >= min_good_samples
            % Interpolate artifacts
            interp_vals = interp1(good_idx, segment(good_idx), 1:length(segment), 'linear', 'extrap');

        else

            % ========== Fallback: Neighbor-window interpolation ==========
            if w > 1 && w < num_windows
                % Previous + next window
                prev_start = max(1, (w-2)*step + 1);
                prev_end   = min(prev_start + win_length - 1, num_pts);
                next_start = w * step + 1;
                next_end   = min(next_start + win_length - 1, num_pts);
                prev_seg = cleaned(ch, prev_start:prev_end);
                next_seg = epoch_data(ch, next_start:next_end);
                len = min(length(prev_seg), length(next_seg));
                interp_vals = mean([prev_seg(1:len); next_seg(1:len)], 1);
            elseif w == 1 && num_windows > 1
                % Only next window
                next_start = w * step + 1;
                next_end   = min(next_start + win_length - 1, num_pts);
                interp_vals = epoch_data(ch, next_start:next_end);
            elseif w == num_windows && num_windows > 1
                % Only previous window
                prev_start = max(1, (w-2)*step + 1);
                prev_end   = min(prev_start + win_length - 1, num_pts);
                interp_vals = cleaned(ch, prev_start:prev_end);
            else
                interp_vals = segment;  % No valid fallback
            end
        end

        % --- Sanity check the interpolated window ---
        too_large = any(abs(interp_vals) > max_amp);
        too_spiky = any(abs(diff(interp_vals)) > max_diff);

        if too_large || too_spiky
            if w > 1 && w < num_windows
                prev_start = max(1, (w-2)*step + 1);
                prev_end   = min(prev_start + win_length - 1, num_pts);
                next_start = w * step + 1;
                next_end   = min(next_start + win_length - 1, num_pts);
                prev_seg = cleaned(ch, prev_start:prev_end);
                next_seg = cleaned(ch, next_start:next_end);
                len = min(length(prev_seg), length(next_seg));
                interp_vals = mean([prev_seg(1:len); next_seg(1:len)], 1);
            else
                interp_vals = repmat(median(segment), 1, length(interp_vals));  % fallback
            end
        end

        % Store interpolated/cleaned segment
        padded_vals = zeros(1, win_length);
        padded_vals(1:length(interp_vals)) = interp_vals;
        window_data(ch,:,w) = padded_vals;
    end
end

% ========== Overlap-Add Reconstruction ==========
reconstruction = zeros(num_ch, num_pts);
weight = zeros(1, num_pts);

for w = 1:num_windows
    idx_start = (w - 1) * step + 1;
    idx_end = min(idx_start + win_length - 1, num_pts);
    len = idx_end - idx_start + 1;
    reconstruction(:, idx_start:idx_end) = reconstruction(:, idx_start:idx_end) + window_data(:,1:len,w);
    weight(idx_start:idx_end) = weight(idx_start:idx_end) + 1;
end


% Normalize by number of overlapping contributions
cleaned = reconstruction ./ weight;

% ========== Final safeguard: Interpolate remaining NaNs ==========
for ch = 1:num_ch
    if any(isnan(cleaned(ch,:)))
        nan_idx = isnan(cleaned(ch,:));
        good_idx = find(~nan_idx);
        cleaned(ch,nan_idx) = interp1(good_idx, cleaned(ch,good_idx), find(nan_idx), 'linear', 'extrap');
    end
end


% ========== POST MAD Inter-channel correlation check ==========
% If a channel still bad based on correlation with neighboring channels,
% interpolate with median average of neighboring channels
for ch = 1:num_ch
    neighbors = setdiff(1:num_ch, ch);
    corrs = zeros(1, numel(neighbors));
    for i = 1:numel(neighbors)
        corrs(i) = corr(cleaned(ch,:)', cleaned(neighbors(i), :)');
    end
    [~, top_idx] = maxk(corrs, min(5, numel(corrs)));
    median_corr = median(corrs(top_idx));

    if median_corr < corr_threshold
        % Replace full channel with top neighbor average
        cleaned(ch,:) = median(cleaned(neighbors(top_idx), :), 1);
    end
end


end
