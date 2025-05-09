Algorithm SequenceSearch(nstage):
    # Initialization
    state_dict = empty dictionary  # Key: sequence state, Value: best completion time
    search_stack = empty stack
    initial_sequence = [''] * nstage
    search_stack.push(initial_sequence)
    
    # Depth-First Search
    while search_stack is not empty:
        current_sequence = search_stack.pop()
        
        # Calculate completion time for the current sequence
        current_time = calculate_completion_time(current_sequence)
        
        # Get state key for pruning
        state_key = get_state_key(current_sequence)
        
        # Pruning: Skip if a better time exists for this state
        if state_key in state_dict and state_dict[state_key] <= current_time:
            continue  # Skip suboptimal path
        
        # Update dictionary with current time
        state_dict[state_key] = current_time
        
        # Generate all valid successor sequences
        successor_sequences = generate_valid_successors(current_sequence)
        
        # Push successors onto the stack
        for new_sequence in successor_sequences:
            search_stack.push(new_sequence)
    
    return best result from state_dict

Function calculate_completion_time(sequence):
    # Compute completion time based on problem-specific logic
    return time

Function get_state_key(sequence):
    # Convert sequence to a unique state representation (e.g., operation counts)
    return state_key

Function generate_valid_successors(sequence):
    successors = empty list
    for each possible operation append:
        new_sequence = copy of sequence
        append operation to new_sequence
        if new_sequence is valid:
            successors.append(new_sequence)
    return successors