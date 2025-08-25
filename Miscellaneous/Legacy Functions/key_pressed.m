function isPressed = key_pressed(target_key)

% If experiment can safely stop the EEG processing while paused, and you dont need real-time responsiveness during the pause, input() is fine and simpler.
% If running a live system where EEG is coming in continuously and might want to pause analysis but keep the stream intact, 
% then the figure-based key polling (like get(gcf, 'CurrentCharacter')) gives more control without blocking the system.

    isPressed = false;

    % Don't trigger anything if no figure exists
    if isempty(findall(0, 'Type', 'figure'))
        fig = figure('Name', 'Calibration Key Listener', ...
            'NumberTitle', 'off', ...
            'MenuBar', 'none', ...
            'ToolBar', 'none', ...
            'Position', [100 100 200 100]);  % Small window
        set(fig, 'Color', 'w');  % white background
    end

    if ~isempty(get(gcf, 'CurrentCharacter'))  % Check if a key was pressed
        key = get(gcf, 'CurrentCharacter');
        if strcmp(key, target_key)
            isPressed = true;
        end
        set(gcf, 'CurrentCharacter', char(0));  % Safely reset key state
    end
end
