import uuid
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import torch
from megatron.core.pipeline_parallel.zerobubble.scheduler import zb, zbv, zbv_greedy,basic1f1b, run_schedule_passes
from megatron.core.parallel_state import get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_rank
from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import F, B, W

@dataclass
class SchedulerConfig:
    n_stages: int
    n_micro: int
    f_cost: List[float]
    b_cost: List[float]
    w_cost: List[float]
    comm_cost: float
    f_mem: List[int]
    b_mem: List[int]
    w_mem: List[int]
    max_mem: List[int]

@dataclass
class ScheduleResult:
    strategy_name: str
    schedule: List[List[Any]]


class ZeroBubbleSchedulerWrapper:
    def __init__(self):
        self.strategies = {
            "basic1f1b": self._run_1f1b,
            "zb_manual": self._run_zb_manual,
            "zbv_vschedule": self._run_zbv_vschedule,
            "zbv_greedy_vmin": self._run_zbv_greedy_vmin,
            "zbv_greedy_vhalf": self._run_zbv_greedy_vhalf,
            "custom_strategy": self._run_custom_strategy
        }

    def _create_graph_config(self, config: SchedulerConfig) -> zb.GraphConfig:
        """Create a GraphConfig for zb.py and other strategies."""
        return zb.GraphConfig(
            cost_f=config.f_cost,
            cost_b=config.b_cost,
            cost_w=config.w_cost,
            cost_comm=config.comm_cost,
            mem_f=config.f_mem,
            mem_b=config.b_mem,
            mem_w=config.w_mem,
            max_mem=config.max_mem,
            print_scaling=1000,
            max_chunks=2 if "zbv" in self.strategies else 1,
            n_stages=config.n_stages,
            n_micro=config.n_micro
        )
    def _run_1f1b(self, config: SchedulerConfig) -> ScheduleResult:
        """Run the basic1f1b.py 1F1B scheduling strategy."""
        graph_config = self._create_graph_config(config)
        local_order = basic1f1b.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="basic_1f1b",
            schedule=schedule,
            completion_time=completion_time,
            bubble_time=bubble_time
        )

    def _run_zb_manual(self, config: SchedulerConfig) -> ScheduleResult:
        """Run the zb.py manual scheduling strategy."""
        graph_config = self._create_graph_config(config)
        local_order = zb.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        #completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="zb_manual",
            schedule=schedule,

        )

    def _run_zbv_vschedule(self, config: SchedulerConfig) -> ScheduleResult:
        """Run the zbv.py v-schedule strategy."""
        f_mid = sum(config.f_cost) / len(config.f_cost)
        b_mid = sum(config.b_cost) / len(config.b_cost)
        w_mid = sum(config.w_cost) / len(config.w_cost)
        graph_config = self._create_graph_config(config)
        pp_graph = zbv.PipelineGraph(
            config.n_stages,
            config.n_micro,
            f_mid, b_mid, w_mid, config.comm_cost,
            f_mem=sum(config.f_mem) / len(config.f_mem),
            b_mem=sum(config.b_mem) / len(config.b_mem),
            w_mem=sum(config.w_mem) / len(config.w_mem),
            max_mem=None
        )
        local_order = pp_graph.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="zbv_vschedule",
            schedule=schedule,
            completion_time=completion_time,
            bubble_time=bubble_time
        )

    def _run_zbv_greedy_vmin(self, config: SchedulerConfig) -> ScheduleResult:
        """Run the zbv_greedy.py v-min strategy."""
        f_mid = sum(config.f_cost) / len(config.f_cost)
        b_mid = sum(config.b_cost) / len(config.b_cost)
        w_mid = sum(config.w_cost) / len(config.w_cost)
        graph_config = self._create_graph_config(config)
        pp_graph = zbv_greedy.PipelineGraph(
            config.n_stages,
            config.n_micro,
            "min",
            int(f_mid), int(b_mid), int(w_mid),
            int(config.comm_cost)
        )
        local_order = pp_graph.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="zbv_greedy_vmin",
            schedule=schedule,
            completion_time=completion_time,
            bubble_time=bubble_time
        )

    def _run_zbv_greedy_vhalf(self, config: SchedulerConfig) -> ScheduleResult:
        """Run the zbv_greedy.py v-half strategy."""
        f_mid = sum(config.f_cost) / len(config.f_cost)
        b_mid = sum(config.b_cost) / len(config.b_cost)
        w_mid = sum(config.w_cost) / len(config.w_cost)
        graph_config = self._create_graph_config(config)
        pp_graph = zbv_greedy.PipelineGraph(
            config.n_stages,
            config.n_micro,
            "half",
            int(f_mid), int(b_mid), int(w_mid),
            int(config.comm_cost)
        )
        local_order = pp_graph.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="zbv_greedy_vhalf",
            schedule=schedule,
            completion_time=completion_time,
            bubble_time=bubble_time
        )

    def _run_custom_strategy(self, config: SchedulerConfig) -> ScheduleResult:
        """Placeholder for a custom scheduling strategy."""
        
        # TODO: Implement your custom scheduling logic here
        # This should return a schedule compatible with run_schedule_passes
        graph_config = self._create_graph_config(config)
        # Example placeholder: use a simple 1F1B schedule as a baseline
        local_order = zb.basic1f1b.create_schedule(graph_config)
        schedule = run_schedule_passes(graph_config, local_order, validate=False)
        completion_time, bubble_time = self._compute_times(schedule, graph_config)
        return ScheduleResult(
            strategy_name="custom_strategy",
            schedule=schedule,
            completion_time=completion_time,
            bubble_time=bubble_time
        )

    def _compute_times(self, schedule: List[List[Any]], config: zb.GraphConfig) -> Tuple[float, float]:
        """Compute theoretical completion time and bubble time for a schedule."""
        stage_times = [0.0] * config.n_stages
        node_times = {}
        rank = 0

        for stage in range(config.n_stages):
            current_time = 0.0
            for node in schedule[stage]:
                # Determine start time based on dependencies
                start_time = current_time
                if node.type == F:
                    # Depends on previous stage's F
                    if stage > 0:
                        prev_f_id = f"F.{node.microbatch}.{stage-1}.{node.seq_split_idx}"
                        start_time = max(start_time, node_times.get(prev_f_id, 0.0) + config.cost_comm)
                    cost = config.cost_f[stage]
                elif node.type == B:
                    # Depends on next stage's B and same stage's F
                    if stage < config.n_stages - 1:
                        next_b_id = f"B.{node.microbatch}.{stage+1}.{node.seq_split_idx}"
                        start_time = max(start_time, node_times.get(next_b_id, 0.0) + config.cost_comm)
                    same_f_id = f"F.{node.microbatch}.{stage}.{node.seq_split_idx}"
                    start_time = max(start_time, node_times.get(same_f_id, 0.0))
                    cost = config.cost_b[stage]
                elif node.type == W:
                    # Depends on same stage's B
                    same_b_id = f"B.{node.microbatch}.{stage}.{node.seq_split_idx}"
                    start_time = max(start_time, node_times.get(same_b_id, 0.0))
                    cost = config.cost_w[stage]
                else:
                    continue

                current_time = start_time + cost
                node_id = f"{node.type}.{node.microbatch}.{stage}.{node.seq_split_idx}"
                node_times[node_id] = current_time
                stage_times[stage] = max(stage_times[stage], current_time)

        completion_time = max(stage_times)
        expected_time = sum(config.cost_f) * config.n_micro + sum(config.cost_b) * config.n_micro + sum(config.cost_w) * config.n_micro
        bubble_time = completion_time - expected_time
        return completion_time, bubble_time

    def _format_schedule(self, schedule: List[List[Any]], stage: int) -> str:
        """Format a stage's schedule as a string of operations."""
        ops = []
        for node in schedule[stage]:
            if node.type in (F, B, W):
                ops.append(str(node.type))
        return "".join(ops)

    def compare_schedules(self, config: SchedulerConfig, strategies: List[str] = None) -> List[ScheduleResult]:
        """Compare schedules and theoretical times for specified strategies."""
        if strategies is None:
            strategies = list(self.strategies.keys())
        results = []
        rank =0
        
        for strategy_name in strategies:
            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            result = self.strategies[strategy_name](config)
            if rank == 0:
                print(f"\nStrategy: {result.strategy_name}")
                for stage in range(config.n_stages):
                    print(f"Stage {stage} Schedule: {self._format_schedule(result.schedule, stage)}")
                
            results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = SchedulerConfig(
        n_stages=4,
        n_micro=8,
        f_cost=[7446.0,6521.0,17203.0,8989.0],
        b_cost=[11266.,10226.0,11436.0,12659.] ,
        w_cost=[0.]*4 ,#[506.0, 514.0,701.0, 578.],
        comm_cost=108.0,
        f_mem=[100] * 4,
        b_mem=[0] * 4,
        w_mem=[-100] * 4,
        max_mem=[200000] * 4
    )
    
    wrapper = ZeroBubbleSchedulerWrapper()
    results = wrapper.compare_schedules(config, strategies=["zb_manual",])