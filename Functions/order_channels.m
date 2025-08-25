function reordered_epoch = order_channels(eeg_epoch)
% REORDER_NAUTILUS_TO_EMOTIV - Rearranges a 14xN EEG epoch from Nautilus
% to match the Emotiv channel order:
% [AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]
% where O1 and O2 are approximated using PO7 and PO8 respectively.

% Check input size
if size(eeg_epoch,1) ~= 14
    error('Input EEG epoch must have 14 channels.');
end

% Define mapping from Nautilus to Emotiv order
% Emotiv target: [AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]
% Nautilus:      [AF3, AF4, F7, F3, F4, F8, FC5, FC6, T7, T8, P7, P8, PO7, PO8]

% Mapping (Emotiv target → Nautilus index):
%  AF3  →  1
%  F7   →  3
%  F3   →  4
%  FC5  →  7
%  T7   →  9
%  P7   → 11
%  O1   → 13   (PO7)
%  O2   → 14   (PO8)
%  P8   → 12
%  T8   → 10
%  FC6  →  8
%  F4   →  5
%  F8   →  6
%  AF4  →  2

nautilus_to_emotiv_idx = [...
    1,  ... AF3 → becomes row 1 in output
    3,  ... F7  → becomes row 2 in output
    4,  ... F3  → becomes row 3, etc.
    7,  ... FC5
    9,  ... T7
    11,  ... P7
    13,  ... O1 (from PO7)
    14,  ... O2 (from PO8)
    12,  ... P8
    10,  ... T8
    8,  ... FC6
    5,  ... F4
    6,  ... F8
    2]; ... AF4

n_channels = numel(nautilus_to_emotiv_idx);
n_samples = size(eeg_epoch, 2);
reordered_epoch = zeros(n_channels, n_samples);

for i = 1:n_channels
    reordered_epoch(i, :) = eeg_epoch(nautilus_to_emotiv_idx(i), :);
end

end

