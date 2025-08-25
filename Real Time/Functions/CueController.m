classdef CueController < handle
    properties
        condition = "LOW";      % Current cue state ("LOW" or "HIGH")
        stopFlag = false;       % Flag to stop the loop
        figHandle               % Handle to the figure for drawing
        timerObj                % Handle parallel figure + code execution
        countdown               % Reset the Countdown every new Cue
        remainingTime           % Countdown as an integer
    end

    methods
        function obj = CueController(figHandle)
            obj.figHandle = figHandle;
        end

        function startTimer(obj)
            if ~isempty(obj.timerObj) && isvalid(obj.timerObj)
                stop(obj.timerObj);
                delete(obj.timerObj);
            end

            obj.timerObj = timer(...
                'ExecutionMode', 'fixedSpacing', ...
                'Period', 0.1, ...
                'BusyMode', 'drop', ...
                'TimerFcn', @(~,~) cue_timer_tick(obj));

            start(obj.timerObj);
        end

        function resetCountdown(obj)            % Reset Timer Countdown
            if strcmpi(obj.condition, 'HIGH')
                obj.countdown = 5;  % 5 sec for high
            else
                obj.countdown = 7;  % 7 sec for low
            end
        end

        function resetInternalTimer(obj)        % Reset Condition for each Block to restart Timer
            % Force internal reset on next tick
            assignin('base', 'cue_timer_reset_flag', true);
        end

        function showMessage(obj, msg, duration)    % Increase User Experience by showing a ADAPT message
            clf(obj.figHandle);
            ax = axes('Parent', obj.figHandle);
            axis off;

            % Show message centered
            text(ax, 0.5, 0.5, msg, ...
                'FontSize', 68, ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Interpreter', 'none');

            drawnow;
            pause(duration);
        end


    end
end
