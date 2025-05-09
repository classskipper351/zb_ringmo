from dataclasses import dataclass
from typing import List, Set

import torch
import pulp

from megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig, ScheduledNode, FuncType


@dataclass
class Graph:
    nstages: int
    nmb: int
    nnodes: int
    config: GraphConfig
    parents: List[Set[int]] = None
    name: List[str] = None
    precede: torch.Tensor = None

    # ID mapping:
    # F[stage][minibatch]: 0..STAGE* MB
    # B[stage][minibatch]: STAGE* MB .. 2 * STAGE * MB
    # W[stage][minibatch]: 2 * STAGE* MB .. 3 * STAGE * MB

    def get_id(self, type, stage, mb):
        return type * (self.nstages * self.nmb) + stage * self.nmb + mb

    def get_stage(self, id):
        return (id // self.nmb) % self.nstages

    def get_cost(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.cost_f[stage], self.config.cost_b[stage], self.config.cost_w[stage]][type]

    def get_mem(self, id):
        type = id // (self.nstages * self.nmb)
        stage = self.get_stage(id)
        return [self.config.mem_f[stage], self.config.mem_b[stage], self.config.mem_w[stage]][type]

    def requires_order(self, i, j):
        return (
            i != j
            and not self.precede[i][j]
            and not self.precede[j][i]
            and self.get_stage(i) == self.get_stage(j)
        )

    @classmethod
    def build_graph(cls, nstages, nmb, config):
        nnodes = nstages * nmb * 3
        g = Graph(nstages=nstages, nmb=nmb, nnodes=nnodes, config=config)
        parents = []
        name = []
        for type in range(3):
            for stage in range(nstages):
                for mb in range(nmb):
                    p = set()
                    if type == 0:
                        name.append(f'F{mb}')#如果 stage > 0（不是第一个阶段），依赖于前一个阶段的相同 microbatch 的 F 节点。
                                            #如果 mb > 0（不是第一个 microbatch），依赖于当前阶段的前一个 microbatch 的 F 节点。
                        if stage > 0:
                            p.add(g.get_id(type, stage - 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 1:#如果 stage == nstages - 1（最后一个阶段，这里是 stage 1），依赖于当前阶段的 F 节点。
                                    #如果不是最后一个阶段，依赖于后一个阶段的相同 microbatch 的 B 节点。
                                    #如果 mb > 0，依赖于当前阶段的前一个 microbatch 的 B 节点。
                        name.append(f'B{mb}')
                        if stage == nstages - 1:
                            p.add(g.get_id(0, stage, mb))
                        else:
                            p.add(g.get_id(type, stage + 1, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    elif type == 2:
                        #依赖于当前阶段和 microbatch 的 B 节点。
                        # 如果 mb > 0，依赖于当前阶段的前一个 microbatch 的 W 节点。
                        name.append(f'W{mb}')
                        p.add(g.get_id(1, stage, mb))
                        if mb > 0:
                            p.add(g.get_id(type, stage, mb - 1))
                    else:
                        assert False
                    parents.append(p)

        g.name = name
        g.parents = parents
        return g

    # Manual ordering producing this kind of schedule:
    # fffffffbfbfbfbfbfbwbwbwbwbwbwbwwwwww
    #  fffffbfbfbfbfbfbfbfbwbwbwbwbwwwwwwww
    #   fffbfbfbfbfbfbfbfbfbfbwbwbwwwwwwwwww
    #    fbfbfbfbfbfbfbfbfbfbfbfbwwwwwwwwwwww
    # Returns the order index of each node on its own stage
    def manual_order(
        self, allow_bubble_before_first_b=False, prioritize_b=False, no_bubble_greedy=True
    ):
        order = [0] * self.nnodes
        f = [0] * self.nstages
        b = [0] * self.nstages
        w = [0] * self.nstages
        o = [0] * self.nstages
        m = [0] * self.nstages
        e = [0] * self.nstages
        t = [0] * self.nnodes
        #max_mem 是每个阶段的内存上限。如果 self.config.max_mem 已定义，则使用配置值；否则，计算每个阶段第一个 F 操作的内存需求（self.get_mem(self.get_id(0, stage, 0))），乘以微批次数量 self.nmb 和一个因子 3（可能是假设每个微批次需要三倍内存来存储中间结果）。这是一个列表，长度为 self.nstages。
        max_mem = self.config.max_mem or [
            self.get_mem(self.get_id(0, stage, 0)) * self.nmb * 3 for stage in range(self.nstages)]
        comm = self.config.cost_comm
        order_str = [""] * self.nstages
        stage_bubble = [0] * self.nstages

        def get_max_bubble():
            max_bubble = 0
            for bb in stage_bubble:
                max_bubble = max(max_bubble, bb)
            return max_bubble

        def put(stage_j, type_k):
            #根据 type_k，从计数器数组 f（前向）、b（后向）或 w（权重更新）中获取当前 microbatch 的索引 _i。
            if type_k == 0:
                _i = f[stage_j]
            elif type_k == 1:
                _i = b[stage_j]
            else:
                _i = w[stage_j]
            _j = stage_j
            _id = self.get_id(type_k, _j, _i)
            _mem = self.get_mem(_id)
            _cost = self.get_cost(_id)
            # TODO
            # assert m[_j] + _mem <= max_mem[stage_j]

            tmp = e[_j] + _cost #初始完成时间：tmp = e[_j] + _cost，其中 e[_j] 是该阶段当前的结束时间。
            no_bubble = tmp
            if _j > 0 and type_k == 0: #如果是 F 操作（type_k == 0）且 stage_j > 0，需要等待前一阶段相同 microbatch 的 F 操作完成并加上通信成本
                tmp = max(tmp, t[self.get_id(0, _j - 1, _i)] + comm + _cost)
            if _j < self.nstages - 1 and type_k == 1:#如果是 B 操作（type_k == 1）且 stage_j < nstages - 1，需要等待后一阶段相同 microbatch 的 B 操作完成并加上通信成本
                tmp = max(tmp, t[self.get_id(1, _j + 1, _i)] + comm + _cost)
            if f[_j] > 0:
                stage_bubble[_j] += tmp - no_bubble #如果该阶段已有 F 操作（f[_j] > 0），计算因依赖导致的额外等待时间（气泡时间）：tmp - no_bubble，并累加到 stage_bubble[_j]。
            #更新阶段结束时间：e[_j] = tmp。
            #记录操作完成时间：t[_id] = tmp。
            #加内存使用：m[_j] += _mem。
            #记录操作顺序：order[_id] = o[_j]。
            #更新对应操作计数器（f[_j]、b[_j] 或 w[_j]）和阶段总操作数 o[_j]。
            #将操作类型（f、b 或 w）追加到 order_str[stage_j]
            e[_j] = tmp
            t[_id] = tmp
            m[_j] += _mem
            order[_id] = o[_j]
            if type_k == 0:
                f[_j] += 1
            elif type_k == 1:
                b[_j] += 1
            else:
                w[_j] += 1
            o[_j] += 1
            fbw = "fbw"
            order_str[stage_j] += fbw[type_k]

        for i in range(self.nmb):
            if i == 0:#这段代码处理第一个微批次（i=0）的调度逻辑，主要分为三步：调度所有阶段的 F0、尝试插入额外的 F 操作、调度 B0。以下是详细分析：
                for j in range(self.nstages):
                    put(j, 0)# 调度所有阶段的 F0
                    # 后续逻辑尝试插入更多 F 操作并调度 B 操作
                f_required = [0] * self.nstages#f_required 是一个列表，长度为 self.nstages，表示每个阶段在调度 B0 前需要额外插入的 F 操作数量，初始化为 0
                last_t = 0 #last_t 表示前一个阶段的完成时间，初始化为 0，用于计算时间依赖。
                for j in range(self.nstages - 1, -1, -1):
                    if j == self.nstages - 1:
                        last_t = t[self.get_id(0, j, i)] + self.get_cost(self.get_id(1, j, i))
                        # j == self.nstages - 1，直接计算 F0 和 B0 的完成时间：t[self.get_id(0, j, i)] 是 F0 的完成时间，加上 B0 的执行成本 self.get_cost(self.get_id(1, j, i))，赋值给 last_t，然后跳过后续逻辑。
                        continue
                    mem = m[j]#mem = m[j]：记录当前内存使用。
                    cost = e[j]#cost = e[j]：记录当前阶段的结束时间。
                    while True:#循环插入 F 操作
                        f_id = self.get_id(0, j, f[j] + f_required[j])#下一个 F 操作的 ID。
                        if f[j] + f_required[j] < self.nmb and mem + self.get_mem(f_id) <= max_mem[j]:
                            if allow_bubble_before_first_b:
                                if cost + self.get_cost(f_id) > last_t + comm:
                                    break
                            else:
                                if cost >= last_t + comm:
                                    break
                            mem += self.get_mem(f_id)
                            cost += self.get_cost(f_id)
                            f_required[j] += 1
                        else:
                            break
                    last_t = max(cost, last_t + comm) + self.get_cost(self.get_id(1, j, i))#取当前阶段结束时间和前一阶段开始时间的最大值，加上 B0 的成本。
                for j in range(self.nstages):
                    while j > 0 and f_required[j] > 0 and f_required[j] >= f_required[j - 1] and f[j] + f_required[
                        j] < self.nmb:
                        f_required[j] -= 1#正序遍历阶段，确保每个阶段的 f_required 不超过前一阶段（避免依赖问题），同时不超过总微批次数。
                for j in range(self.nstages):
                    for _ in range(f_required[j]):
                        put(j, 0)#根据 f_required[j]，调用 put(j, 0) 插入额外的 F 操作，填补气泡。
                for j in range(self.nstages - 1, -1, -1):
                    put(j, 1)#逆序遍历阶段，调用 put(j, 1) 调度每个阶段的第一个后向传播操作（B0）。
                continue
            # 确定每个阶段是否需要 F 操作
            f_required = [0] * self.nstages
            for j in range(self.nstages):
                if f[j] >= self.nmb:#该阶段的 F 操作已全部调度，跳过。
                    continue
                if j + 1 < self.nstages and f[j] >= f[j + 1] + 2 and prioritize_b: #如果 prioritize_b 为真且 f[j] >= f[j + 1] + 2（当前阶段 F 操作领先下一阶段至少 2 个），计算下一阶段的下一次 F 和 B 操作完成时间（next_plus_fw）。
                    next_plus_fw = (
                        e[j + 1]
                        + self.get_cost(self.get_id(0, j + 1, f[j + 1]))
                        + self.get_cost(self.get_id(1, j + 1, b[j + 1]))
                        + comm
                    )
                    if e[j] >= next_plus_fw:
                        continue
                    f_id = self.get_id(0, j, f[j])
                    f_mem = self.get_mem(f_id)
                    w_cost, w_cnt = 0, 0
                    mem_with_w = m[j] + f_mem
                    while mem_with_w > max_mem[j] and w[j] + w_cnt < b[j]:
                        w_id = self.get_id(2, j, w[j] + w_cnt)
                        w_cost += self.get_cost(w_id)
                        mem_with_w += self.get_mem(w_id)
                        w_cnt += 1
                    if e[j] + self.get_cost(f_id) + w_cost <= next_plus_fw:
                        f_required[j] = 1
                        continue

                    w_cost, w_cnt = 0, 0
                    # mem_with_w = m[j]
                    # while w[j] + w_cnt < b[j]:
                    #     w_id = self.get_id(2, j, w[j] + w_cnt)
                    #     w_cost += self.get_cost(w_id)
                    #     mem_with_w += self.get_mem(w_id)
                    #     w_cnt += 1
                    # if e[j] + w_cost >= next_plus_fw:
                    #     continue
                    if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                        # TODO: can sample here
                        continue
                f_required[j] = 1
            for j in range(self.nstages - 2, -1, -1):
                f_required[j] = min(f_required[j], f_required[j + 1])#逆序调整 f_required，确保当前阶段的 F 操作数量不超过下一阶段（维持依赖关系）。
            for j in range(self.nstages):
                if f_required[j] == 0:
                    continue
                f_id = self.get_id(0, j, f[j])
                mem = self.get_mem(f_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j > 0:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(0, j - 1, f[j])] + comm
                    ):
                        put(j, 2)#非第一阶段：如果 j > 0，检查是否能在等待前一阶段 F 操作的时间内插入 W 操作：
                    if w[j] < b[j] and e[j] < t[self.get_id(0, j - 1, f[j])] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(0, j - 1, f[j])] + comm
                        if (
                            t[self.get_id(0, j - 1, f[j])] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                put(j, 0)
            for j in range(self.nstages - 1, -1, -1):
                assert b[j] == i
                b_id = self.get_id(1, j, b[j])
                mem = self.get_mem(b_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory") #如果 W 操作已用尽（w[j] >= b[j]），抛出内存不足错误。连一个B都放不下。
                    put(j, 2)
                if j + 1 < self.nstages:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(1, j + 1, i)] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(1, j + 1, i)] + comm:
                        # TODO: e[j] + self.get_cost(self.get_id(2, j, w[j])) > t[self.get_id(1, j + 1, i)] + comm
                        if (
                            t[self.get_id(1, j + 1, i)] + comm - e[j]
                            <= get_max_bubble() - stage_bubble[j]
                        ):
                            # TODO: can sample here
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                if j == 0 and f[j] == self.nmb:
                    while w[j] < b[j]:
                        put(j, 2)
                put(j, 1)

        for i in range(self.nstages):
            while w[i] < self.nmb:
                put(i, 2)#确保所有 W 操作被调度。
            # print(f"{' ' * i}{order_str[i]}  -> {e[i]}")

        for i in range(self.nstages):
            for j in range(self.nmb):
                f_id = self.get_id(0, i, j)
                b_id = self.get_id(1, i, j)
                w_id = self.get_id(2, i, j)
                f_cost = self.get_cost(f_id)
                b_cost = self.get_cost(b_id)
                w_cost = self.get_cost(w_id)
                assert t[b_id] >= t[f_id] + b_cost # b 依赖f
                assert t[w_id] >= t[b_id] + w_cost, f"{i}-{j}, {t[w_id]} >= {t[b_id]} + {w_cost}" # w依赖b
                if i > 0:
                    assert t[f_id] >= t[self.get_id(0, i - 1, j)] + comm + f_cost, f"{i}-{j}"
                if i < self.nstages - 1:
                    assert t[b_id] >= t[self.get_id(1, i + 1, j)] + comm + b_cost

        # print(order)
        best_time = 0
        for i in range(self.nstages):
            time_i = (
                t[self.get_id(2, i, self.nmb - 1)]
                - t[self.get_id(0, i, 0)]
                + self.get_cost(self.get_id(0, i, 0))
            )
            best_time = max(best_time, time_i)
        bubble_rates = []
        for i in range(self.nstages):
            time_i = (
                t[self.get_id(2, i, self.nmb - 1)]
                - t[self.get_id(0, i, 0)]
                + self.get_cost(self.get_id(0, i, 0))
            )
            bubble_rate_i = stage_bubble[i] / time_i if time_i > 0 else 0
            bubble_rates.append(bubble_rate_i)
            #print(f"Stage {i} Bubble Rate: {bubble_rate_i:.4f} (Bubble Time: {stage_bubble[i]}, Total Time: {time_i})")

        return order, t, best_time , bubble_rates


def initial_solution(graph, print_result=True):
    best_time, order, complete_time ,final_br= None, None, None,None
    for allow_bubble_before_first_b in [True, False]:
        for prioritize_b in [True, False]:
            for no_bubble_greedy in [True, False]:
                order_t, complete_time_t, best_time_t , bubble_rates = graph.manual_order(
                    allow_bubble_before_first_b=allow_bubble_before_first_b,
                    prioritize_b=prioritize_b,
                    no_bubble_greedy=no_bubble_greedy,
                )
                if best_time is None or best_time_t < best_time:
                    best_time = best_time_t
                    order = order_t
                    complete_time = complete_time_t
                    final_br = bubble_rates
                    
    print(f"bubble_rates: {final_br}")
    print(f"averatge_bubble_rates:{sum(final_br)/len(final_br)}")
    
                    

    if 1 :
        print_detail(graph, complete_time)
        print("-" * 20, best_time, "-" * 20)
    return best_time, order, complete_time


def print_detail(graph, F):
    typenames = ['F', 'B', 'W']
    times = []
    for stage in range(graph.nstages):
        stage_str = ['.'] * int(F[graph.get_id(2, stage, graph.nmb - 1)] / graph.config.print_scaling)
        for _type in range(3):
            for _mb in range(graph.nmb):
                _id = graph.get_id(_type, stage, _mb)
                end = int(F[_id] / graph.config.print_scaling)
                start = int((F[_id] - graph.get_cost(_id)) / graph.config.print_scaling)
                for j in range(start, end):
                    if j == start or j == end - 1:
                        stage_str[j] = typenames[_type]
                    elif j == start + 1:
                        if _mb >= 10:
                            stage_str[j] = str(_mb // 10)
                        else:
                            stage_str[j] = str(_mb)
                    elif j == start + 2 and _mb >= 10:
                        stage_str[j] = str(_mb % 10)
                    else:
                        stage_str[j] = "-"
        _str = ""
        for _c in stage_str:
            _str += _c
        times.append(
            F[graph.get_id(2, stage, graph.nmb - 1)]
            - F[graph.get_id(0, stage, 0)]
            + graph.get_cost(graph.get_id(0, stage, 0))
        )
        print(_str)
    print('Longest stage time: ', max(times))


def create_schedule(config: GraphConfig, print_result=False):
    graph = Graph.build_graph(config.n_stages, config.n_micro, config)
    best_time, order, complete_time = initial_solution(graph, print_result)
    return create_scheduled_nodes(graph, complete_time)


def create_scheduled_nodes(graph, completion_time):
    typenames = [FuncType.F, FuncType.B, FuncType.W]
    cats = {
        FuncType.F: 0,
        FuncType.B: 1,
        FuncType.W: 2,
    }
    local_order = []
    end_time = []
    for t in completion_time:
        end_time.append(pulp.value(t))
    for stage in range(graph.nstages):
        order = []
        for cat in range(3):
            for mb in range(graph.nmb):
                order.append(
                    ScheduledNode(
                        type=typenames[cat],
                        stage=stage,
                        microbatch=mb,
                        layer_group_idx=stage,
                    )
                )
        order = sorted(order, key=lambda n: completion_time[graph.get_id(cats[n.type], n.stage, n.microbatch)])
        local_order.append(order)
    return local_order
