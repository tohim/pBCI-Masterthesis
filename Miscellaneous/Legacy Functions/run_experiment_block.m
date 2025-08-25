% -------------------------------------------------------------------------
% REAL TIME PARADIGM FUNCTION
% -------------------------------------------------------------------------
function run_experiment_block(block_idx, workload_label, udp_handle, mode, state)
% Determine command from Label (Labels get shuffled before the
% Real-Time Loop Starts - this is then Input for this Paradigm function)

switch lower(state)

    case 'realtime'

        if workload_label == 0
            command = "LOW";
        else
            command = "HIGH";
        end

        fprintf('\n=== Starting Block %d | MWL = %s | Mode = %s ===\n', ...
            block_idx, initial_command, upper(mode));

        % Send command to robot
        send_robot_command(udp_handle, command);
        assignin('base', 'ground_truth_label', workload_label);

        block_duration = 30;
        correction_time = 20;
        pause_step = 1;
        t_start = tic;

        already_corrected = false;

        while toc(t_start) < block_duration
            pause(pause_step);

            % If we run the mode "adaptive":
            % After 20 seconds if no correction was done yet - send
            % correction_command based on final_MWL

            if strcmpi(mode, 'adaptive') && ~already_corrected && toc(t_start) >= correction_time
                try
                    final_MWL = evalin('base', 'final_MWL');
                    if final_MWL == 0
                        correction_command = "LOW";
                    else
                        correction_command = "HIGH";
                    end

                    fprintf('[ADAPT] At %.1f s → MWL=%d → Sending correction: %s\n', ...
                        toc(t_start), final_MWL, correction_command);

                    send_robot_command(udp_handle, correction_command);
                    already_corrected = true;  % Only correct once

                catch
                    warning('[ADAPT] Could not read final_MWL from base workspace.');
                end
            end
        end

        send_robot_command(udp_handle, "STOP");
        fprintf('Block complete. Robot resting for 10 seconds...\n');
        pause(10);

    case 'testing'

        % simulate at least mode "fixed" - maybe if possible also mode
        % "adaptive". 
        % but at least we can start with building a simulation
        % "run_experiment_block" so that we can test a continous eeg
        % dataflow and to make predictions and storing these predictions
        % if this works, then we know at least, that we could send the
        % correct command

end

end