
from dataclasses import dataclass
from typing import List
import uuid

@dataclass
class Operation:
    type: str  # 'F' or 'B' (B includes W)
    stage: int
    microbatch: int
    start_time: float
    end_time: float

def calculate_1f1b_bubble_rates(n_stages: int, n_microbatches: int, cost_f: List[float], 
                              cost_b: List[float], cost_w: List[float], comm_cost: float) -> tuple[List[float], float]:
    """
    Calculate bubble rates for each stage and average bubble rate using 1F1B scheduling.
    
    Args:
        n_stages: Number of pipeline stages
        n_microbatches: Number of microbatches
        cost_f: List of forward operation times for each stage
        cost_b: List of backward operation times for each stage
        cost_w: List of weight update times for each stage
        comm_cost: Communication cost between stages
    
    Returns:
        tuple: (bubble_rates, avg_bubble_rate)
            - bubble_rates: List of bubble rates for each stage
            - avg_bubble_rate: Average bubble rate across all stages
    """
    # Initialize state variables
    schedules = [[] for _ in range(n_stages)]  # Operation schedule for each stage
    end_times = [0.0] * n_stages  # Current end time for each stage
    bubble_times = [0.0] * n_stages  # Accumulated bubble time for each stage
    total_times = [0.0] * n_stages  # Total execution time for each stage

    # 1F1B Scheduling
    for mb in range(n_microbatches):
        # Schedule F operations for microbatch mb
        for stage in range(n_stages):
            # Calculate expected start time (without dependencies)
            expected_start = end_times[stage]
            actual_start = expected_start

            # Check F dependency: depends on previous stage's F for same microbatch
            if stage > 0:
                prev_f_end = next((op.end_time for op in schedules[stage-1] 
                                  if op.type == 'F' and op.microbatch == mb), 0)
                actual_start = max(actual_start, prev_f_end + comm_cost)

            # Record bubble time
            bubble_time = actual_start - expected_start
            bubble_times[stage] += bubble_time

            # Schedule F operation
            op_end_time = actual_start + cost_f[stage]
            op = Operation(type='F', stage=stage, microbatch=mb, 
                          start_time=actual_start, end_time=op_end_time)
            schedules[stage].append(op)
            end_times[stage] = op_end_time

        # Schedule B operations (including W) for microbatch mb
        for stage in range(n_stages-1, -1, -1):
            # Calculate expected start time
            expected_start = end_times[stage]
            actual_start = expected_start

            # Check B dependency: depends on next stage's B or current stage's F
            if stage < n_stages - 1:
                next_b_end = next((op.end_time for op in schedules[stage+1] 
                                  if op.type == 'B' and op.microbatch == mb), 0)
                actual_start = max(actual_start, next_b_end + comm_cost)
            else:
                # For last stage, B depends on its own F
                curr_f_end = next((op.end_time for op in schedules[stage] 
                                  if op.type == 'F' and op.microbatch == mb), 0)
                actual_start = max(actual_start, curr_f_end)

            # Record bubble time
            bubble_time = actual_start - expected_start
            bubble_times[stage] += bubble_time

            # Schedule B operation (including W)
            op_end_time = actual_start + cost_b[stage] + cost_w[stage]
            op = Operation(type='B', stage=stage, microbatch=mb, 
                          start_time=actual_start, end_time=op_end_time)
            schedules[stage].append(op)
            end_times[stage] = op_end_time

    # Calculate total execution time and bubble rate for each stage
    bubble_rates = []
    for stage in range(n_stages):
        # Total time: from first F start to last B end
        first_f_start = next((op.start_time for op in schedules[stage] 
                             if op.type == 'F' and op.microbatch == 0), 0)
        last_b_end = next((op.end_time for op in schedules[stage] 
                          if op.type == 'B' and op.microbatch == n_microbatches-1), 0)
        total_time = last_b_end - first_f_start
        total_times[stage] = total_time

        # Bubble rate: bubble time / total time
        bubble_rate = bubble_times[stage] / total_time if total_time > 0 else 0
        bubble_rates.append(bubble_rate)

    # Calculate average bubble rate
    avg_bubble_rate = sum(bubble_rates) / n_stages if n_stages > 0 else 0

    # Print results
    for stage in range(n_stages):
        print(f"Stage {stage}:")
        print(f"  Total Time: {total_times[stage]:.2f}")
        print(f"  Bubble Time: {bubble_times[stage]:.2f}")
        print(f"  Bubble Rate: {bubble_rates[stage]:.4f}")
    print(f"Average Bubble Rate: {avg_bubble_rate:.4f}")

    return bubble_rates, avg_bubble_rate

# Example usage
if __name__ == "__main__":
    # Example inputs
    n_stages = 4
    n_microbatches = 8
    cost_f = [8135,6600,6757,9493 ]  # Forward times for each stage
    cost_b = [7309,6708,11259,10451]  # Backward times for each stage
    cost_w = [0,0,0,0]  # Weight update times for each stage
    comm_cost = 238  # Communication cost

    bubble_rates, avg_bubble_rate = calculate_1f1b_bubble_rates(
        n_stages, n_microbatches, cost_f, cost_b, cost_w, comm_cost
    )
    
