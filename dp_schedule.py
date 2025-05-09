from collections import defaultdict, deque
import copy

def is_valid_state(sequences, nmb, nstages, ignore_w=False):
    """验证当前序列是否合法"""
    if len(sequences) != nstages:
        return False
    for seq in sequences:
        try:
            # 根据 ignore_w 设置最大序列长度
            max_len = nmb * (2 if ignore_w else 3)
            assert len(seq) <= max_len
            # 检查操作类型
            valid_ops = ['f', 'b'] if ignore_w else ['f', 'b', 'w']
            for op in seq:
                assert op in valid_ops, "包含非法操作"
            # 检查 F 和 B 的计数
            for op in ['f', 'b']:
                assert seq.count(op) <= nmb
            # 如果不忽略 W，检查 W 的计数
            if not ignore_w:
                assert seq.count('w') <= nmb
            # 检查 F -> B -> W 依赖
            F_stack = deque()
            B_stack = deque()
            for op in seq:
                if op == 'f':
                    F_stack.append(op)
                elif op == 'b':
                    assert F_stack, "B 操作前必须有 F"
                    B_stack.append(op)
                    F_stack.pop()
                elif op == 'w' and not ignore_w:
                    assert B_stack, "W 操作前必须有 B"
                    B_stack.pop()
        except AssertionError:
            return False
    for stage in range(nstages - 1):
        prev_seq = sequences[stage]
        succ_seq = sequences[stage + 1]
        try:
            assert prev_seq.count('f') >= succ_seq.count('f')
            assert prev_seq.count('b') <= succ_seq.count('b')
        except AssertionError:
            return False
    return True

def find_successors(sequences, nmb, nstages, ignore_w=False):
    """生成当前序列的所有合法后继"""
    successors = []
    for stage in range(nstages):
        # 根据 ignore_w 设置可添加的操作
        ops = ['f', 'b'] if ignore_w else ['f', 'b', 'w']
        for op in ops:
            new_seq = [list(s) for s in sequences]
            new_seq[stage].append(op)
            new_seq_str = [''.join(s) for s in new_seq]
            if is_valid_state(new_seq_str, nmb, nstages, ignore_w):
                successors.append(new_seq_str)
    return successors

def get_fbw_counts(sequences, ignore_w=False):
    """将序列转换为 FBW 操作数元组"""
    return tuple(
        (seq.count('f'), seq.count('b'), 0 if ignore_w else seq.count('w'))
        for seq in sequences
    )

def get_local_time(f, b, w, c, nmb, nstages, sequences, ignore_w=False):
    """计算当前序列的完成时间"""
    seq_len = [len(seq) for seq in sequences]
    seq_index = [0] * nstages
    curr_time = [0.0] * nstages
    f_queues = [deque() for _ in range(nstages)]
    b_queues = [deque() for _ in range(nstages)]
    
    def get_dep(stage, op):
        if op == 'f':
            return stage - 1 if stage > 0 else None
        elif op == 'b':
            return stage + 1 if stage < nstages - 1 else None
        return None
    
    while any(seq_index[s] < seq_len[s] for s in range(nstages)):
        for stage in range(nstages):
            if seq_index[stage] >= seq_len[stage]:
                continue
            op = sequences[stage][seq_index[stage]]
            dep_stage = get_dep(stage, op)
            
            if dep_stage is None:
                if op == 'f':
                    curr_time[stage] += f[stage]
                    f_queues[stage].append(curr_time[stage])
                elif op == 'b':
                    curr_time[stage] += b[stage]
                    b_queues[stage].append(curr_time[stage])
                elif op == 'w' and not ignore_w:
                    curr_time[stage] += w[stage]
                seq_index[stage] += 1
            else:
                if op == 'f':
                    if f_queues[dep_stage]:
                        dep_time = f_queues[dep_stage].popleft()
                        curr_time[stage] = max(curr_time[stage], dep_time + c) + f[stage]
                        f_queues[stage].append(curr_time[stage])
                        seq_index[stage] += 1
                elif op == 'b':
                    if b_queues[dep_stage]:
                        dep_time = b_queues[dep_stage].popleft()
                        curr_time[stage] = max(curr_time[stage], dep_time + c) + b[stage]
                        b_queues[stage].append(curr_time[stage])
                        seq_index[stage] += 1
    return max(curr_time)

def global_sequence_search(nstage, nmb, f, b, w, c, ignore_w=False):
    """全局搜索最优调度序列"""
    global_best_time = float('inf')
    best_schedule = []
    
    # 状态字典：记录 FBW 操作数元组 -> 最优完成时间
    state_dict = defaultdict(lambda: float('inf'))
    
    # 初始状态：所有 stage 为空序列
    initial_sequences = [''] * nstage
    stack = [(initial_sequences, [])]  # (当前序列, 路径)
    
    while stack:
        current_sequences, path = stack.pop()
        
        # 计算当前序列的完成时间
        current_time = get_local_time(f, b, w, c, nmb, nstage, current_sequences, ignore_w)
        
        # 获取 FBW 操作数元组作为状态键
        fbw_key = get_fbw_counts(current_sequences, ignore_w)
        
        # 剪枝：如果当前时间不优于已有记录，跳过
        if current_time >= state_dict[fbw_key]:
            continue
        state_dict[fbw_key] = current_time
        
        # 检查是否所有操作完成
        target_len = (2 if ignore_w else 3) * nmb
        if all(len(seq) == target_len for seq in current_sequences):
            if current_time < global_best_time:
                global_best_time = current_time
                best_schedule = path[:]
            continue
        
        # 生成后继，保持使用序列
        successors = find_successors(current_sequences, nmb, nstage, ignore_w)
        for successor in successors:
            new_path = path + [successor]
            stack.append((successor, new_path))
    
    return best_schedule, global_best_time


# 示例用法
if __name__ == "__main__":
    import time
    nstage, nmb = 4, 8
    f = [7446.,6521.,17203.,8989.][:nstage]
    b = [11266.,10226.,11436.,12659.,] [:nstage]
    w =[506.0, 514.0,701.0, 578.] [:nstage]
    c = 108
    start_time = time.time()
    schedule, opt_time = global_sequence_search(nstage, nmb, f, b, w, c ,ignore_w=True)
    end_time = time.time()
    elapsed = end_time - start_time  # 计算耗时（秒）
    print(f"耗时: {elapsed:.4f} 秒")
    print(f"最优调度路径: {schedule}")
    print(f"最优完成时间: {opt_time}")