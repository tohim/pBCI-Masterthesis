function cue_display_loop(ctrl)
    % ctrl: CueController object

    % Define colors
    colors = {'orange', 'yellow', 'blue', 'green'};
    rgb_map = containers.Map({'orange','yellow','blue','green'}, ...
        {[1 0.5 0], [1 1 0], [0 0.5 1], [0 1 0]});

    while ~ctrl.stopFlag
        clf(ctrl.figHandle); hold on;
        axis off;
        rectangle('Position', [0 0 1 1], 'FaceColor', 'white');

        % Get condition and generate cue
        cond = ctrl.condition;
        shuffled = colors(randperm(4));
        if strcmpi(cond, 'HIGH')
            selected = shuffled(1:3);
            duration = 6;
        else
            selected = shuffled(1);
            duration = 8;
        end

        % Draw color cue
        for i = 1:length(selected)
            posY = 0.7 - (i-1)*0.3;
            rectangle('Position',[0.4 posY 0.2 0.2], ...
                      'Curvature',[1 1], ...
                      'FaceColor', rgb_map(selected{i}), ...
                      'EdgeColor','k', 'LineWidth', 2);
        end

        % Timer display
        if strcmpi(cond, 'HIGH')
            for t = duration:-1:1
                if ctrl.stopFlag, break; end
                txt = text(0.05, 0.95, sprintf('%d..', t), 'FontSize', 18, 'FontWeight', 'bold');
                pause(1);
                delete(txt);
            end
        else
            pause(duration);
        end
    end

    clf(ctrl.figHandle); % blank out figure when loop stops
end
