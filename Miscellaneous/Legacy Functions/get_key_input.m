function key = get_key_input()
    key = '';
    while isempty(key)
        pause(0.1);
        key = get(gcf, 'CurrentCharacter'); % Read the current key press
    end
    set(gcf, 'CurrentCharacter', char(0));  % Safely reset key state
end