function cue_timer_tick(ctrl)
    persistent t counter colors rgb_map selected ax

    if ctrl.stopFlag
        return
    end

    if isempty(t)
        t = 0; counter = 0; selected = {};
        colors = {'orange','yellow','blue','green'};
        rgb_map = containers.Map(colors, {[1 0.5 0], [1 1 0], [0 0.5 1], [0 1 0]});
        ax = axes('Parent', ctrl.figHandle);  % assign axis to cue figure
    end

    % Handle block reset (force refresh even for same condition)
    if evalin('base','exist(''cue_timer_reset_flag'',''var'')') && evalin('base','cue_timer_reset_flag')
        t = 0;
        counter = 0;
        shuffled = colors(randperm(4));
        if strcmpi(ctrl.condition, 'HIGH')
            selected = shuffled(1:3);
        else
            selected = shuffled(1);
        end
        evalin('base','clear cue_timer_reset_flag');
    end

    % Redraw every few frames
    if counter == 0
        clf(ctrl.figHandle);
        ax = axes('Parent', ctrl.figHandle, 'Position', [0 0 1 1]);  % Full figure coverage
        axis(ax, [0 1 0 1]);          % Logical coordinate space
        axis(ax, 'off');
        axis(ax, 'equal');            % Prevent stretching

        % Background
        rectangle(ax, 'Position', [0 0 1 1], 'FaceColor', 'white', 'EdgeColor', 'none');

        % Color Circles
        for i = 1:length(selected)
            posY = 0.7 - (i-1)*0.3;
            rectangle(ax, 'Position', [0.4 posY 0.2 0.2], ...
                      'Curvature', [1 1], ...
                      'FaceColor', rgb_map(selected{i}), ...
                      'EdgeColor', 'k', ...
                      'LineWidth', 2, ...
                      'Parent', ax);
        end
    end

    % Draw countdown number every full second
    duration = strcmpi(ctrl.condition, 'HIGH') * 5 + ...
               strcmpi(ctrl.condition, 'LOW') * 7;

    if mod(counter, 10) == 0
        delete(findall(ctrl.figHandle, 'Type', 'text'));  % clear old timer text
        remaining = duration - t;
        if remaining >= 0
            text(ax, 0.05, 0.95, sprintf('%d', round(remaining)), ...
                 'FontSize', 48, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
        end
    end

    % Increment time
    counter = mod(counter + 1, 10);
    t = t + 0.1;

    % Reset cue visuals if timer exceeded
    if t >= duration
        t = 0; counter = 0;
        shuffled = colors(randperm(4));
        if strcmpi(ctrl.condition, 'HIGH')
            selected = shuffled(1:3);
        else
            selected = shuffled(1);
        end
    end
end
